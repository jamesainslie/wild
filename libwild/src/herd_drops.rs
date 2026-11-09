//! Drops values that we allocate in a `bumpalo_herd::Herd`. A herd frees its memory without running
//! the destructors of what it holds, so any heap memory that those values own would leak. A process
//! that links once and exits doesn't notice, but one that links repeatedly grows on every link.

use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::PoisonError;

/// Slices of values, allocated in a herd, that need dropping before the herd frees them.
#[derive(Debug, Default)]
pub(crate) struct HerdDrops {
    slices: Mutex<Vec<ErasedSlice>>,
}

#[derive(Debug)]
struct ErasedSlice {
    ptr: NonNull<()>,
    len: usize,
    drop: unsafe fn(NonNull<()>, usize),
}

// SAFETY: `adopt` only accepts slices of `Send` values, so they can be dropped from any thread.
unsafe impl Send for ErasedSlice {}

impl HerdDrops {
    /// Takes over dropping `values` and returns them as a shared slice.
    ///
    /// # Safety
    ///
    /// - The memory behind `values` must stay allocated until `run` is called or `self` is dropped.
    /// - Nothing else may drop the values.
    /// - Dropping a `T` must not read anything that `T` borrows. `run` may execute after those
    ///   borrows have ended, since erasing `T` hides them from the borrow checker.
    pub(crate) unsafe fn adopt<'a, T: Send>(&'a self, values: &'a mut [T]) -> &'a [T] {
        let len = values.len();
        let ptr = NonNull::from(values).cast::<T>();
        self.slices
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(ErasedSlice {
                ptr: ptr.cast(),
                len,
                drop: drop_slice::<T>,
            });
        // SAFETY: `ptr` came from a unique reference that's valid for 'a. We only use it again in
        // `run`, which takes `&mut self` and so can't be called until 'a has ended.
        unsafe { NonNull::slice_from_raw_parts(ptr, len).as_ref() }
    }

    /// Drops everything that we've adopted.
    pub(crate) fn run(&mut self) {
        let slices = self
            .slices
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        for slice in slices.drain(..) {
            // SAFETY: `adopt` recorded `drop` together with the slice's type, and its caller
            // promised that the slice is still allocated and that nothing else drops it.
            unsafe { (slice.drop)(slice.ptr, slice.len) };
        }
    }
}

impl Drop for HerdDrops {
    fn drop(&mut self) {
        self.run();
    }
}

/// # Safety
///
/// `ptr` and `len` must describe an allocated slice of `T` that nothing else drops.
unsafe fn drop_slice<T>(ptr: NonNull<()>, len: usize) {
    // SAFETY: Our caller upholds the requirements of `drop_in_place`.
    unsafe { NonNull::slice_from_raw_parts(ptr.cast::<T>(), len).drop_in_place() };
}

#[cfg(test)]
mod tests {
    use super::HerdDrops;
    use std::sync::Arc;

    #[test]
    fn run_drops_adopted_values_once() {
        let herd = bumpalo_herd::Herd::new();
        let value = Arc::new(());
        let mut drops = HerdDrops::default();
        {
            let member = herd.get();
            let values = member.alloc_slice_fill_iter([Arc::clone(&value), Arc::clone(&value)]);
            // SAFETY: `herd` outlives `drops`, and nothing else drops the values.
            let values = unsafe { drops.adopt(values) };
            assert_eq!(values.len(), 2);
        }
        assert_eq!(Arc::strong_count(&value), 3);
        drops.run();
        assert_eq!(Arc::strong_count(&value), 1);
        drops.run();
        assert_eq!(Arc::strong_count(&value), 1);
    }
}
