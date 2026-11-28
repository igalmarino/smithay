//! Shm buffers uploaded through pixel unpack buffers of the renderer's own
//! (GLES 3), kept and reused.
//!
//! Given client memory, `glTexSubImage2D` has the driver copy it into a
//! staging buffer it allocates: on radeonsi a new buffer object whose pages the
//! kernel faults in as the copy first touches them, so a whole 2560x1440
//! buffer cost 6.5 to 7.5 ms of CPU on a Renoir APU and, when the pages had to
//! be found anew, 13 to 22 ms. Copied instead into a buffer whose pages were
//! touched once ahead (`GlesRenderer::reserve_shm_uploads`), the copy is a
//! plain `memcpy`, ~2.4 ms for the same buffer, and the GPU copies it into the
//! texture. The first upload this way still paid the driver's one-time work
//! for the path (9 ms on that APU's bare metal), so reserving also makes a
//! tiny one through each buffer.
//!
//! Two buffers take turns, each with the fence of the upload that last read
//! it, so a copy rarely waits for the GPU to be done with the previous one.

use std::time::{Duration, Instant};

use tracing::debug;

use super::ffi::{self, Gles2};
use crate::utils::{Buffer as BufferCoord, Rectangle};

/// An upload slower than this logs where its time went.
const SLOW: Duration = Duration::from_millis(4);

/// Uploads smaller than this go straight from client memory, in strips: the
/// driver's own staging is cheap at that size.
pub(super) const MIN_BYTES: usize = 256 * 1024;

#[derive(Debug)]
struct Slot {
    pbo: ffi::types::GLuint,
    capacity: usize,
    /// The upload that last read the buffer, until the GPU is done with it.
    fence: Option<ffi::types::GLsync>,
}

/// The renderer's pixel unpack buffers for shm uploads.
#[derive(Debug, Default)]
pub(super) struct ShmUploads {
    slots: [Option<Slot>; 2],
    next: usize,
    /// Whether a first upload went through them (`GlesRenderer::reserve_shm_uploads`).
    pub(super) warmed: bool,
}

impl ShmUploads {
    /// Make both buffers hold at least `bytes`, their pages touched once:
    /// the one-time cost a first upload of that size would otherwise pay.
    ///
    /// # Safety
    ///
    /// The renderer's context is current.
    pub(super) unsafe fn reserve(&mut self, gl: &Gles2, bytes: usize) -> bool {
        let reserved = (0..self.slots.len()).all(|index| unsafe { self.slot(gl, index, bytes).is_some() });
        unsafe { gl.BindBuffer(ffi::PIXEL_UNPACK_BUFFER, 0) };
        reserved
    }

    /// The buffer at `index`, done with by the GPU and holding at least
    /// `bytes`, bound to `PIXEL_UNPACK_BUFFER`; `None` if it can't be made.
    unsafe fn slot(&mut self, gl: &Gles2, index: usize, bytes: usize) -> Option<&mut Slot> {
        let slot = self.slots[index].get_or_insert_with(|| {
            let mut pbo = 0;
            unsafe { gl.GenBuffers(1, &mut pbo) };
            Slot {
                pbo,
                capacity: 0,
                fence: None,
            }
        });
        if slot.pbo == 0 {
            return None;
        }
        unsafe {
            gl.BindBuffer(ffi::PIXEL_UNPACK_BUFFER, slot.pbo);
            if let Some(fence) = slot.fence.take() {
                gl.ClientWaitSync(fence, ffi::SYNC_FLUSH_COMMANDS_BIT, u64::MAX);
                gl.DeleteSync(fence);
            }
            if slot.capacity < bytes {
                let size = isize::try_from(bytes).ok()?;
                gl.GetError();
                gl.BufferData(ffi::PIXEL_UNPACK_BUFFER, size, std::ptr::null(), ffi::STREAM_DRAW);
                let map = gl.MapBufferRange(ffi::PIXEL_UNPACK_BUFFER, 0, size, ffi::MAP_WRITE_BIT);
                if map.is_null() || gl.GetError() != ffi::NO_ERROR {
                    slot.capacity = 0;
                    gl.BindBuffer(ffi::PIXEL_UNPACK_BUFFER, 0);
                    return None;
                }
                // Touched once now, so an upload's copy finds its pages.
                std::ptr::write_bytes(map as *mut u8, 0, bytes);
                gl.UnmapBuffer(ffi::PIXEL_UNPACK_BUFFER);
                slot.capacity = bytes;
            }
        }
        Some(slot)
    }

    /// Upload `regions` of the buffer at `pixels` (`stride` bytes a row,
    /// `pixelsize` bytes a pixel) to the bound texture through the next
    /// buffer; false if it couldn't, and nothing was uploaded.
    ///
    /// # Safety
    ///
    /// The renderer's context is current, the texture is bound, and `pixels`
    /// holds every region at that stride.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn upload(
        &mut self,
        gl: &Gles2,
        regions: &[Rectangle<i32, BufferCoord>],
        pixels: *const u8,
        stride: usize,
        pixelsize: usize,
        format: ffi::types::GLenum,
        type_: ffi::types::GLenum,
    ) -> bool {
        let bytes: usize = regions
            .iter()
            .map(|r| r.size.w as usize * r.size.h as usize * pixelsize)
            .sum();
        if bytes == 0 {
            return true;
        }
        let index = self.next;
        self.next = (self.next + 1) % self.slots.len();
        let started = Instant::now();
        let waited = self.slots[index]
            .as_ref()
            .is_some_and(|slot| slot.fence.is_some());
        let grown = self.slots[index]
            .as_ref()
            .is_none_or(|slot| slot.capacity < bytes);
        let Some(slot) = (unsafe { self.slot(gl, index, bytes) }) else {
            return false;
        };
        let ready = started.elapsed();
        unsafe {
            let map = gl.MapBufferRange(
                ffi::PIXEL_UNPACK_BUFFER,
                0,
                bytes as isize,
                ffi::MAP_WRITE_BIT | ffi::MAP_UNSYNCHRONIZED_BIT | ffi::MAP_INVALIDATE_RANGE_BIT,
            ) as *mut u8;
            if map.is_null() {
                gl.BindBuffer(ffi::PIXEL_UNPACK_BUFFER, 0);
                return false;
            }
            let mapped = started.elapsed();
            // Each region packed, its rows `row` bytes apart.
            let mut offset = 0;
            let mut placed = Vec::with_capacity(regions.len());
            for region in regions {
                let row = region.size.w as usize * pixelsize;
                let first = region.loc.y as usize * stride + region.loc.x as usize * pixelsize;
                if row == stride {
                    std::ptr::copy_nonoverlapping(
                        pixels.add(first),
                        map.add(offset),
                        row * region.size.h as usize,
                    );
                } else {
                    for y in 0..region.size.h as usize {
                        std::ptr::copy_nonoverlapping(
                            pixels.add(first + y * stride),
                            map.add(offset + y * row),
                            row,
                        );
                    }
                }
                placed.push((*region, offset));
                offset += row * region.size.h as usize;
            }
            let copied = started.elapsed();
            gl.UnmapBuffer(ffi::PIXEL_UNPACK_BUFFER);
            let unmapped = started.elapsed();
            for (region, offset) in placed {
                gl.PixelStorei(ffi::UNPACK_ROW_LENGTH, region.size.w);
                gl.TexSubImage2D(
                    ffi::TEXTURE_2D,
                    0,
                    region.loc.x,
                    region.loc.y,
                    region.size.w,
                    region.size.h,
                    format,
                    type_,
                    offset as *const _,
                );
            }
            slot.fence = Some(gl.FenceSync(ffi::SYNC_GPU_COMMANDS_COMPLETE, 0));
            gl.BindBuffer(ffi::PIXEL_UNPACK_BUFFER, 0);
            let total = started.elapsed();
            if total > SLOW {
                let ms = |d: Duration| d.as_secs_f64() * 1000.0;
                debug!(
                    bytes,
                    slot = index,
                    waited,
                    grown,
                    ready_ms = ms(ready),
                    map_ms = ms(mapped - ready),
                    copy_ms = ms(copied - mapped),
                    unmap_ms = ms(unmapped - copied),
                    texture_ms = ms(total - unmapped),
                    "slow shm upload"
                );
            }
        }
        true
    }

    /// Delete the buffers and fences.
    ///
    /// # Safety
    ///
    /// The renderer's context is current.
    pub(super) unsafe fn destroy(&mut self, gl: &Gles2) {
        for slot in self.slots.iter_mut().filter_map(Option::take) {
            unsafe {
                if let Some(fence) = slot.fence {
                    gl.DeleteSync(fence);
                }
                gl.DeleteBuffers(1, &slot.pbo);
            }
        }
    }
}
