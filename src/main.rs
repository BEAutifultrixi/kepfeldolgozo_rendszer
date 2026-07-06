use clap::Parser;
use bytemuck;
use wgpu::util::DeviceExt;

#[derive(Parser, Debug)]
#[command(author, version, about = "Rust párhuzamos képfeldolgozó CLI")]
struct Args {
    /// A feldolgozandó kép elérési útja
    #[arg(short, long, default_value = "bemenet.jpg")]
    image: String,

    /// Alkalmazandó szűrő (negativ, szurke, fenyero, blur, sobel, all)
    #[arg(short, long, default_value = "all")]
    filter: String,

    /// Futtatási mód (cpu vagy gpu)
    #[arg(short, long, default_value = "cpu")]
    mode: String,
}

use image::{GenericImageView, ImageBuffer, Rgb, Luma};
use rayon::prelude::*;
use std::fs::File; // Fájlkezelés beimportálása a mentéshez
use std::io::Write; // Írási műveletek beimportálása
use std::time::Instant;

fn main() {
    let args = Args::parse();
    
    // 1. Kép beolvasása a parancssorból kapott elérési úttal
    let img = match image::open(&args.image) {
        Ok(file) => file,
        Err(_) => {
            println!("Hiba: A '{}' nevű kép nem található!", args.image);
            return;
        }
    };

    let (width, height) = img.dimensions();
    println!("=== KÉPFELDOLGOZÁS MÉRÉSI JELENTÉS ===");
    println!("Kép felbontása: {}x{} pixel (Összesen: {} pixel)", width, height, width * height);
    println!("Kiválasztott szűrő: {}", args.filter);
    println!("Futtatási mód: {}\n", args.mode);

    // Fájl előkészítése a mérési adatoknak (létrehozom a CSV fájlt)
    let mut csv_file = File::create("meresek.csv").expect("Nem sikerült létrehozni a CSV fájlt");
    // CSV Fejléc írása (Oszlopnevek pontosvesszővel elválasztva)
    writeln!(csv_file, "Algoritmus;Szekvencialis (ms);Parhuzamos (ms);Gyorsulas").unwrap();

    // --- 1. ALGORITMUS: NEGATÍV EFFEKT ---
    if args.filter == "negativ" || args.filter == "all" {
        println!("--- 1. Algoritmus: Negatív effekt ---");
        let mut pixels_seq = img.to_rgb8().into_raw();
        let mut pixels_par = pixels_seq.clone();
        let mut pixels_gpu = pixels_seq.clone();

        // 1. A: Szekvenciális CPU
        let start = Instant::now();
        for pixel in pixels_seq.chunks_mut(3) {
            pixel[0] = 255 - pixel[0];
            pixel[1] = 255 - pixel[1];
            pixel[2] = 255 - pixel[2];
        }
        let duration_seq_neg = start.elapsed().as_secs_f64() * 1000.0;
        println!("Szekvenciális (1 mag CPU) futási idő: {:.2} ms", duration_seq_neg);

        // 1. B: Párhuzamos CPU (Rayon)
        let start = Instant::now();
        pixels_par.par_chunks_mut(3).for_each(|pixel| {
            pixel[0] = 255 - pixel[0];
            pixel[1] = 255 - pixel[1];
            pixel[2] = 255 - pixel[2];
        });
        let duration_par_neg = start.elapsed().as_secs_f64() * 1000.0;
        println!("Párhuzamos (Több mag CPU) futási idő:  {:.2} ms", duration_par_neg);
        let speedup_neg = duration_seq_neg / duration_par_neg;
        println!("CPU Gyorsulás mértéke: {:.2}x", speedup_neg);

        // 1. C: GPU (wgpu Compute Shader) -- Csak ha a mód 'gpu' vagy az összeset tesztelem 'all'
        let mut duration_gpu_neg = 0.0;
        if args.mode == "gpu" || args.mode == "all" {
            let start = Instant::now();
            if let Some(gpu_result) = pollster::block_on(run_negativ_gpu(&pixels_gpu)) {
                pixels_gpu = gpu_result;
                duration_gpu_neg = start.elapsed().as_secs_f64() * 1000.0;
                println!("GPU gyorsított futási idő: {:.2} ms", duration_gpu_neg);
                println!("GPU Gyorsulás a szekvenciálishoz: {:.2}x", duration_seq_neg / duration_gpu_neg);
            } else {
                println!("GPU hiba: Nem sikerült inicializálni a hardveres gyorsítást.");
            }
        }
        println!();
        
        // Mérési adatok mentése a fájlba
        writeln!(csv_file, "Negativ effekt CPU;{:.2};{:.2};{:.2}", duration_seq_neg, duration_par_neg, speedup_neg).unwrap();
        if duration_gpu_neg > 0.0 {
            writeln!(csv_file, "Negativ effekt GPU;{:.2};{:.2};{:.2}", duration_seq_neg, duration_gpu_neg, duration_seq_neg / duration_gpu_neg).unwrap();
        }

        // Kép mentése (ha GPU módban vagyok, a GPU által visszaadott bufferből mentek)
        let final_pixels = if args.mode == "gpu" { &pixels_gpu } else { &pixels_par };
        if let Some(output_img) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, final_pixels.to_vec()) {
            output_img.save("kimenet_negativ.jpg").unwrap();
        }
    }

    // --- 2. ALGORITMUS: SZÜRKEÁRNYALATOSÍTÁS (SÚLYOZOTT) ---
    if args.filter == "szurke" || args.filter == "all" {
        println!("--- 2. Algoritmus: Szürkeárnyalatosítás ---");
        let mut pixels_seq = img.to_rgb8().into_raw();
        let mut pixels_par = pixels_seq.clone();
        let mut pixels_gpu = pixels_seq.clone();

        // Szekvenciális
        let start = Instant::now();
        for pixel in pixels_seq.chunks_mut(3) {
            let gray = (pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114) as u8;
            pixel[0] = gray;
            pixel[1] = gray;
            pixel[2] = gray;
        }
        let duration_seq_gray = start.elapsed().as_secs_f64() * 1000.0;
        println!("Szekvenciális (1 mag) futási idő: {:.2} ms", duration_seq_gray);

        // Párhuzamos (Rayon)
        let start = Instant::now();
        pixels_par.par_chunks_mut(3).for_each(|pixel| {
            let gray = (pixel[0] as f32 * 0.299 + pixel[1] as f32 * 0.587 + pixel[2] as f32 * 0.114) as u8;
            pixel[0] = gray;
            pixel[1] = gray;
            pixel[2] = gray;
        });
        let duration_par_gray = start.elapsed().as_secs_f64() * 1000.0;
        println!("Párhuzamos (Több mag) futási idő:  {:.2} ms", duration_par_gray);
        let speedup_gray = duration_seq_gray / duration_par_gray;
        println!("CPU Gyorsulás mértéke: {:.2}x", speedup_gray);

        // GPU gyorsított (wgpu) -- Csak ha 'gpu' mód vagy 'all' van kiválasztva
        let mut duration_gpu_gray = 0.0;
        if args.mode == "gpu" || args.mode == "all" {
            let start = Instant::now();
            if let Some(gpu_result) = pollster::block_on(run_szurke_gpu(&pixels_gpu)) {
                pixels_gpu = gpu_result;
                duration_gpu_gray = start.elapsed().as_secs_f64() * 1000.0;
                println!("GPU gyorsított futási idő:            {:.2} ms", duration_gpu_gray);
                println!("GPU Gyorsulás a szekvenciálishoz:    {:.2}x", duration_seq_gray / duration_gpu_gray);
            } else {
                println!("GPU hiba: Nem sikerült inicializálni a hardveres gyorsítást.");
            }
        }
        println!();
        
        // Szürkeárnyalatosítás adatainak elmentése a fájlba
        writeln!(csv_file, "Szürkeányalatosítás CPU;{:.2};{:.2};{:.2}", duration_seq_gray, duration_par_gray, speedup_gray).unwrap();
        if duration_gpu_gray > 0.0 {
            writeln!(csv_file, "Szürkeányalatosítás GPU;{:.2};{:.2};{:.2}", duration_seq_gray, duration_gpu_gray, duration_seq_gray / duration_gpu_gray).unwrap();
        }

        let final_pixels = if args.mode == "gpu" { &pixels_gpu } else { &pixels_par };
        if let Some(output_img) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, final_pixels.to_vec()) {
            output_img.save("kimenet_szurke.jpg").unwrap();
        }
    }

    // --- 3. ALGORITMUS: FÉNYERŐ NÖVELÉSE ---
    if args.filter == "fenyero" || args.filter == "all" {
        println!("--- 3. Algoritmus: Fényerő növelése ---");
        let mut pixels_seq = img.to_rgb8().into_raw();
        let mut pixels_par = pixels_seq.clone();

        // Szekvenciális
        let start = Instant::now();
        for pixel in pixels_seq.chunks_mut(3) {
            pixel[0] = pixel[0].saturating_add(50);
            pixel[1] = pixel[1].saturating_add(50);
            pixel[2] = pixel[2].saturating_add(50);
        }
        let duration_seq_light = start.elapsed().as_secs_f64() * 1000.0;
        println!("Szekvenciális (1 mag) futási idő: {:.2} ms", duration_seq_light);

        // Párhuzamos (Rayon)
        let start = Instant::now();
        pixels_par.par_chunks_mut(3).for_each(|pixel| {
            pixel[0] = pixel[0].saturating_add(50);
            pixel[1] = pixel[1].saturating_add(50);
            pixel[2] = pixel[2].saturating_add(50);
        });
        let duration_par_light = start.elapsed().as_secs_f64() * 1000.0;
        println!("Párhuzamos (Több mag) futási idő:  {:.2} ms", duration_par_light);
        let speedup_light = duration_seq_light / duration_par_light;
        println!("CPU Gyorsulás mértéke: {:.2}x\n", speedup_light);

        // Adat mentése a CSV fájlba
        writeln!(csv_file, "Fényerő növelése CPU;{:.2};{:.2};{:.2}", duration_seq_light, duration_par_light, speedup_light).unwrap();

        if let Some(output_img) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, pixels_par) {
            output_img.save("kimenet_fenyero.jpg").unwrap();
        }
    }

    // --- 4. ALGORITMUS: SOBEL ÉLKERESÉS ---
    if args.filter == "sobel" || args.filter == "all" {
        println!("--- 4. Algoritmus: Sobel élkeresés ---");
        
        let gray_img = img.to_luma8();
        let gray_raw = gray_img.as_raw();
        let mut pixels_seq = vec![0u8; (width * height) as usize];
        let mut pixels_par = vec![0u8; (width * height) as usize];
        let mut pixels_gpu = vec![0u8; (width * height) as usize];

        // Szekvenciális CPU
        let start = Instant::now();
        for y in 1..height-1 {
            for x in 1..width-1 {
                let idx = (y * width + x) as usize;
                let gx = -1.0 * gray_raw[((y-1)*width + x-1) as usize] as f32 + 1.0 * gray_raw[((y-1)*width + x+1) as usize] as f32 
                         -2.0 * gray_raw[(y*width + x-1) as usize] as f32       + 2.0 * gray_raw[(y*width + x+1) as usize] as f32 
                         -1.0 * gray_raw[((y+1)*width + x-1) as usize] as f32 + 1.0 * gray_raw[((y+1)*width + x+1) as usize] as f32;

                let gy = -1.0 * gray_raw[((y-1)*width + x-1) as usize] as f32 - 2.0 * gray_raw[((y-1)*width + x) as usize] as f32 - 1.0 * gray_raw[((y-1)*width + x+1) as usize] as f32 
                         +1.0 * gray_raw[((y+1)*width + x-1) as usize] as f32 + 2.0 * gray_raw[((y+1)*width + x) as usize] as f32 + 1.0 * gray_raw[((y+1)*width + x+1) as usize] as f32;

                pixels_seq[idx] = (gx*gx + gy*gy).sqrt().min(255.0) as u8;
            }
        }
        let duration_seq_sobel = start.elapsed().as_secs_f64() * 1000.0;
        println!("Szekvenciális (1 mag) futási idő: {:.2} ms", duration_seq_sobel);

        // Párhuzamos CPU (Rayon)
        let start = Instant::now();
        pixels_par.par_chunks_mut(width as usize).enumerate().for_each(|(y, row)| {
            if y == 0 || y == (height - 1) as usize { return; }
            for x in 1..(width - 1) as usize {
                let w = width as usize;
                let gx = -1.0 * gray_raw[(y-1)*w + x-1] as f32 + 1.0 * gray_raw[(y-1)*w + x+1] as f32 
                         -2.0 * gray_raw[y*w + x-1] as f32     + 2.0 * gray_raw[y*w + x+1] as f32 
                         -1.0 * gray_raw[(y+1)*w + x-1] as f32 + 1.0 * gray_raw[(y+1)*w + x+1] as f32;

                let gy = -1.0 * gray_raw[(y-1)*w + x-1] as f32 - 2.0 * gray_raw[(y-1)*w + x] as f32 - 1.0 * gray_raw[(y-1)*w + x+1] as f32 
                         +1.0 * gray_raw[(y+1)*w + x-1] as f32 + 2.0 * gray_raw[(y+1)*w + x] as f32 + 1.0 * gray_raw[(y+1)*w + x+1] as f32;

                row[x] = (gx*gx + gy*gy).sqrt().min(255.0) as u8;
            }
        });
        let duration_par_sobel = start.elapsed().as_secs_f64() * 1000.0;
        println!("Párhuzamos (Több mag) futási idő:  {:.2} ms", duration_par_sobel);
        let speedup_sobel = duration_seq_sobel / duration_par_sobel;
        println!("CPU Gyorsulás mértéke: {:.2}x", speedup_sobel);

        // GPU gyorsított (wgpu)
        let mut duration_gpu_sobel = 0.0;
        if args.mode == "gpu" || args.mode == "all" {
            let start = Instant::now();
            if let Some(gpu_result) = pollster::block_on(run_sobel_gpu(gray_raw, width, height)) {
                pixels_gpu = gpu_result;
                duration_gpu_sobel = start.elapsed().as_secs_f64() * 1000.0;
                println!("GPU gyorsított futási idő:            {:.2} ms", duration_gpu_sobel);
                println!("GPU Gyorsulás a szekvenciálishoz:    {:.2}x", duration_seq_sobel / duration_gpu_sobel);
            } else {
                println!("GPU hiba: Nem sikerült inicializálni a hardveres gyorsítást.");
            }
        }
        println!();

        writeln!(csv_file, "Sobel elkereses CPU;{:.2};{:.2};{:.2}", duration_seq_sobel, duration_par_sobel, speedup_sobel).unwrap();
        if duration_gpu_sobel > 0.0 {
            writeln!(csv_file, "Sobel elkereses GPU;{:.2};{:.2};{:.2}", duration_seq_sobel, duration_gpu_sobel, duration_seq_sobel / duration_gpu_sobel).unwrap();
        }

        let final_pixels = if args.mode == "gpu" && duration_gpu_sobel > 0.0 { &pixels_gpu } else { &pixels_par };
        if let Some(output_img) = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, final_pixels.to_vec()) {
            output_img.save("kimenet_sobel.jpg").unwrap();
        }
    }

    // --- 5. ALGORITMUS: BOX BLUR (HOMÁLYOSÍTÁS) ---
    if args.filter == "blur" || args.filter == "all" {
        println!("--- 5. Algoritmus: Box Blur (Homályosítás) ---");
        let raw_pixels = img.to_rgb8().into_raw();
        
        // Szekvenciális konvolúció
        let mut output_seq = vec![0u8; raw_pixels.len()];
        let start = Instant::now();
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut r_sum = 0u32;
                let mut g_sum = 0u32;
                let mut b_sum = 0u32;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = x + kx - 1;
                        let py = y + ky - 1;
                        let idx = ((py * width + px) * 3) as usize;
                        r_sum += raw_pixels[idx] as u32;
                        g_sum += raw_pixels[idx + 1] as u32;
                        b_sum += raw_pixels[idx + 2] as u32;
                    }
                }
                let out_idx = ((y * width + x) * 3) as usize;
                output_seq[out_idx] = (r_sum / 9) as u8;
                output_seq[out_idx + 1] = (g_sum / 9) as u8;
                output_seq[out_idx + 2] = (b_sum / 9) as u8;
            }
        }
        let duration_seq_blur = start.elapsed().as_secs_f64() * 1000.0;
        println!("Szekvenciális (1 mag) futási idő: {:.2} ms", duration_seq_blur);

        // Párhuzamos konvolúció (Rayon sorbontással)
        let mut output_par = vec![0u8; raw_pixels.len()];
        let start = Instant::now();
        output_par.par_chunks_mut((width * 3) as usize)
            .enumerate()
            .for_each(|(y, row)| {
                if y == 0 || y == (height - 1) as usize { return; }
                for x in 1..(width - 1) {
                    let mut r_sum = 0u32;
                    let mut g_sum = 0u32;
                    let mut b_sum = 0u32;

                    for ky in 0..3 {
                        for kx in 0..3 {
                            let px = x + kx - 1;
                            let py = y + ky - 1;
                            let idx = ((py * (width as usize) + px as usize) * 3) as usize;
                            r_sum += raw_pixels[idx] as u32;
                            g_sum += raw_pixels[idx + 1] as u32;
                            b_sum += raw_pixels[idx + 2] as u32;
                        }
                    }
                    let out_x_idx = (x * 3) as usize;
                    row[out_x_idx] = (r_sum / 9) as u8;
                    row[out_x_idx + 1] = (g_sum / 9) as u8;
                    row[out_x_idx + 2] = (b_sum / 9) as u8;
                }
            });
        let duration_par_blur = start.elapsed().as_secs_f64() * 1000.0;
        println!("Párhuzamos (Több mag) futási idő:  {:.2} ms", duration_par_blur);
        let speedup_blur = duration_seq_blur / duration_par_blur;
        println!("Gyorsulás mértéke: {:.2}x\n", speedup_blur);

        // Box Blur adatainak elmentése a fájlba
        writeln!(csv_file, "Box Blur;{:.2};{:.2};{:.2}", duration_seq_blur, duration_par_blur, speedup_blur).unwrap();

        if let Some(output_img_blur) = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, output_par) {
            output_img_blur.save("kimenet_blur.jpg").unwrap();
        }
    }

    println!("Minden mérési adat sikeresen kimentve a 'meresek.csv' fájlba!");
}

// --- GPU COMPUTE BACKEND IMPLEMENTÁCIÓ ---
async fn run_negativ_gpu(input_data: &[u8]) -> Option<Vec<u8>> {
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Negatív Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
            @group(0) @binding(0) var<storage, read_write> pixels: array<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                if (idx >= arrayLength(&pixels)) { return; }
                pixels[idx] = pixels[idx] ^ 0x00FFFFFFu;
            }
        "#)),
    });

    let size = input_data.len() as u64;
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU Tárhely Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Átmeneti CPU Olvasható Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: storage_buffer.as_entire_binding() }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((size / 64 + 1) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if rx.recv().ok()?.is_ok() {
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        Some(result)
    } else {
        None
    }
}

async fn run_szurke_gpu(input_data: &[u8]) -> Option<Vec<u8>> {
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Szürke Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
            @group(0) @binding(0) var<storage, read_write> pixels: array<u32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let idx = id.x;
                if (idx >= arrayLength(&pixels)) { return; }
                
                let packed_pixel = pixels[idx];
                let r = f32(packed_pixel & 0xFFu);
                let g = f32((packed_pixel >> 8u) & 0xFFu);
                let b = f32((packed_pixel >> 16u) & 0xFFu);
                let a = packed_pixel & 0xFF000000u;

                let gray = u32(r * 0.299 + g * 0.587 + b * 0.114);
                pixels[idx] = gray | (gray << 8u) | (gray << 16u) | a;
            }
        "#)),
    });

    let size = input_data.len() as u64;
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU Szürke Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: storage_buffer.as_entire_binding() }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((size / 64 + 1) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if rx.recv().ok()?.is_ok() {
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        Some(result)
    } else {
        None
    }
}

async fn run_sobel_gpu(input_data: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.ok()?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sobel Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
            @group(0) @binding(0) var<storage, read> input_pixels: array<u32>;
            @group(0) @binding(1) var<storage, read_write> output_pixels: array<u32>;
            @group(0) @binding(2) var<storage, read> dimensions: vec2<u32>;

            fn get_pixel(x: u32, y: u32, w: u32, h: u32) -> f32 {
                if (x >= w || y >= h) { return 0.0; }
                let pixel_idx = (y * w + x) / 4u;
                let byte_idx = (y * w + x) % 4u;
                let packed = input_pixels[pixel_idx];
                return f32((packed >> (byte_idx * 8u)) & 0xFFu);
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let w = dimensions.x;
                let h = dimensions.y;
                let x = id.x;
                let y = id.y;

                if (x >= w || y >= h) { return; }
                let out_idx = y * w + x;
                
                if (x == 0u || x == w - 1u || y == 0u || y == h - 1u) {
                    return;
                }

                let gx = -1.0 * get_pixel(x - 1u, y - 1u, w, h) + 1.0 * get_pixel(x + 1u, y - 1u, w, h)
                         -2.0 * get_pixel(x - 1u, y, w, h)      + 2.0 * get_pixel(x + 1u, y, w, h)
                         -1.0 * get_pixel(x - 1u, y + 1u, w, h) + 1.0 * get_pixel(x + 1u, y + 1u, w, h);

                let gy = -1.0 * get_pixel(x - 1u, y - 1u, w, h) - 2.0 * get_pixel(x, y - 1u, w, h) - 1.0 * get_pixel(x + 1u, y - 1u, w, h)
                         +1.0 * get_pixel(x - 1u, y + 1u, w, h) + 2.0 * get_pixel(x, y + 1u, w, h) + 1.0 * get_pixel(x + 1u, y + 1u, w, h);

                let magnitude = u32(sqrt(gx * gx + gy * gy));
                let edge = clamp(magnitude, 0u, 255u);
                
                let p_idx = out_idx / 4u;
                let b_idx = out_idx % 4u;
                
                output_pixels[p_idx] = output_pixels[p_idx] | (edge << (b_idx * 8u));
            }
        "#)),
    });

    let u32_count = (input_data.len() + 3) / 4;
    let size = (u32_count * 4) as u64;

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(input_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dim_data = [width, height];
    let dim_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dim Buffer"),
        contents: bytemuck::cast_slice(&dim_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dim_buffer.as_entire_binding() },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[] });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: Some(&pipeline_layout), module: &shader, entry_point: "main" });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((width / 16 + 1) as u32, (height / 16 + 1) as u32, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if rx.recv().ok()?.is_ok() {
        let data = buffer_slice.get_mapped_range();
        let result: &[u8] = bytemuck::cast_slice(&data);
        let final_res = result[..input_data.len()].to_vec();
        drop(data);
        staging_buffer.unmap();
        Some(final_res)
    } else {
        None
    }
}