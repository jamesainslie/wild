//! Linking the same inputs many times in one process must not grow the heap. A process that keeps
//! running and links again on every change, such as a build tool's watch mode, would otherwise grow
//! by whatever each link fails to free.

// Rayon can't spawn threads on WASI.
#![cfg(not(target_family = "wasm"))]

use libwild::Linker;
use object::write::Object;
use object::write::StandardSection;
use object::write::Symbol;
use object::write::SymbolSection;
use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::cell::Cell;
use std::path::Path;
use std::sync::atomic::AtomicIsize;
use std::sync::atomic::Ordering;

/// Counts the heap allocations, and the bytes in them, that are live, so that the test can see what
/// each link leaves behind. Only the thread that links counts, so that the test harness's own
/// allocations on other threads don't.
struct CountingAllocator;

static LIVE_ALLOCATIONS: AtomicIsize = AtomicIsize::new(0);
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record(allocations: isize, bytes: isize) {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        LIVE_ALLOCATIONS.fetch_add(allocations, Ordering::Relaxed);
        LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}

fn live() -> (isize, isize) {
    (
        LIVE_ALLOCATIONS.load(Ordering::Relaxed),
        LIVE_BYTES.load(Ordering::Relaxed),
    )
}

// SAFETY: Every method forwards to `System`, which upholds `GlobalAlloc`'s contract. The counters
// don't allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Our caller upholds `alloc`'s contract, which is also `System::alloc`'s.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record(1, layout.size() as isize);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: As for `alloc`.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record(1, layout.size() as isize);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Our caller upholds `realloc`'s contract, which is also `System::realloc`'s.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            record(0, new_size as isize - layout.size() as isize);
        }
        new_ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Our caller upholds `dealloc`'s contract, which is also `System::dealloc`'s.
        unsafe { System.dealloc(ptr, layout) };
        record(-1, -(layout.size() as isize));
    }
}

/// An object file that defines a single function, `_start`, in `.text`.
fn create_object() -> Vec<u8> {
    let mut object = Object::new(
        object::BinaryFormat::Elf,
        object::Architecture::X86_64,
        object::Endianness::Little,
    );
    let text = object.section_id(StandardSection::Text);
    let start = object.add_symbol(Symbol {
        name: b"_start".to_vec(),
        value: 0,
        size: 0,
        kind: object::SymbolKind::Text,
        scope: object::SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Undefined,
        flags: object::SymbolFlags::None,
    });
    // ud2
    object.add_symbol_data(start, text, &[0x0f, 0x0b], 1);
    object.write().unwrap()
}

fn link(input: &Path, output: &Path) {
    let arguments = [
        "wild",
        "-flavor",
        "gnu",
        "-m",
        "elf_x86_64",
        "-static",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ];
    let get_arguments = || arguments.into_iter();
    let linker = Linker::new();
    let mut args = libwild::Args::new(get_arguments).unwrap();
    args.parse(get_arguments, linker.file_system()).unwrap();

    let linker_output = linker.run(&args).unwrap();
    drop(linker_output);
    drop(linker);
}

#[test]
fn repeated_links_free_what_they_allocate() {
    const LINKS: usize = 20;

    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("start.o");
    let output = dir.path().join("start");
    std::fs::write(&input, create_object()).unwrap();

    // One thread, so that the pool's per-thread state is the same on every link.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    let (allocations, bytes) = pool.install(|| {
        let link_and_drain = || {
            link(&input, &output);
            // Run any jobs that the link spawned and didn't wait for, so that what they free is
            // freed before we count. Our thread is the pool's only worker.
            while rayon::yield_now() == Some(rayon::Yield::Executed) {}
        };
        // The first link sets up state that lives for the whole process, such as the thread
        // pool's queues, so we count from the end of it.
        link_and_drain();
        COUNTING.set(true);
        for _ in 1..LINKS {
            link_and_drain();
        }
        COUNTING.set(false);
        live()
    });

    assert_eq!(
        (allocations, bytes),
        (0, 0),
        "{allocations} more allocations and {bytes} more bytes live after {LINKS} links than \
         after the first"
    );
}
