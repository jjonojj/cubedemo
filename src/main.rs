mod keymap;

const NAME: &str = "okhaloma";

const SCR_SIZE: (u32, u32) = (360, 270);

use std::f32::consts::PI;

use glam::{Mat4, Vec3};

use image::GenericImageView;
use wgpu::util::DeviceExt;
use wgpu::{InstanceDescriptor, SurfaceError};

use sdl3::event::{Event, WindowEvent};
use sdl3::keyboard::Keycode;

#[allow(dead_code, unused)]
struct Texture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

fn create_depth_texture(device: &wgpu::Device, label: &str) -> Texture {
    let size = wgpu::Extent3d {
        width: SCR_SIZE.0,
        height: SCR_SIZE.1,
        depth_or_array_layers: 1,
    };
    let desc = wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT // 3.
                | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let texture = device.create_texture(&desc);

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        // 4.
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual), // 5.
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        ..Default::default()
    });

    Texture {
        texture,
        view,
        sampler,
    }
}

fn mk_texbg(
    dev: &wgpu::Device,
    q: &wgpu::Queue,
    bytes: &[u8],
    label: &str,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let diff_img = image::load_from_memory(bytes).unwrap();
    let diff_rgba = diff_img.to_rgba8();
    let dimensions = diff_img.dimensions();

    let texsize = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1,
    };

    let tex = dev.create_texture(&wgpu::TextureDescriptor {
        size: texsize,
        label: Some(label),
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        mip_level_count: 1,
        sample_count: 1,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    q.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &diff_rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * dimensions.0),
            rows_per_image: Some(dimensions.1),
        },
        texsize,
    );

    let diffuse_texture_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let diffuse_sampler = dev.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }, wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None
        }],
        label: None,
    });

    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bgl,
        label: None,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
            }
        ]
    });

    (bgl, bg)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

// free camera (player?)
struct Camera {
    eye: Vec3,
    forw: Vec3,
    up: Vec3,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    speed: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub sens: f32,
    pub pitch_max: f32,
}

impl Camera {
    #[rustfmt::skip]
    pub const OPENGL_TO_WGPU_MAT4: Mat4 = Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.5,
        0.0, 0.0, 0.0, 1.0,
    ]);

    pub fn mkvpm(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.eye + self.forw, self.up);
        let proj = Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);
        return Self::OPENGL_TO_WGPU_MAT4 * proj * view;
    }

    fn look(&mut self, dx: f32, dy: f32) {
        self.yaw += dx.to_radians() * self.sens;
        self.pitch -= dy.to_radians() * self.sens;
        if self.yaw > 360f32.to_radians() {
            self.yaw -= 360f32.to_radians();
        }
        self.pitch = self
            .pitch
            .clamp(-self.pitch_max.to_radians(), self.pitch_max.to_radians());
    }

    fn update(&mut self, config: &wgpu::SurfaceConfiguration, keymap: &keymap::Keymap) {
        self.aspect = config.width as f32 / config.height as f32;

        // compute forward vector
        let fvx = self.pitch.cos() * self.yaw.cos();
        let fvy = self.pitch.sin();
        let fvz = self.pitch.cos() * self.yaw.sin();

        self.forw = Vec3::new(fvx, fvy, fvz).normalize();

        let right = Vec3::new(self.forw.x, 0., self.forw.z)
            .normalize()
            .cross(self.up);
        if keymap.get(Keycode::W) {
            self.eye += self.forw * self.speed;
        }
        if keymap.get(Keycode::A) {
            self.eye -= right * self.speed;
        }
        if keymap.get(Keycode::S) {
            self.eye -= self.forw * self.speed;
        }
        if keymap.get(Keycode::D) {
            self.eye += right * self.speed;
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PrepassUniforms {
    delta: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    delta: f32,
    _pad: [f32; 3],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            mvp: Mat4::IDENTITY.to_cols_array_2d(),
            delta: 0.0,
            _pad: [0.0; 3],
        }
    }

    fn update(&mut self, cam: &Camera, model: Mat4) {
        self.mvp = (cam.mkvpm() * model).to_cols_array_2d();
    }
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[rustfmt::skip]
const VERTICES: &[Vertex] = &[
    // Front face
    Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], color: [0.0, 1.0, 0.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 0.0, 1.0] },
    Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 1.0, 0.0] },

    // Back face
    Vertex { position: [-0.5, -0.5, -0.5], color: [1.0, 0.0, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 1.0, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], color: [1.0, 1.0, 1.0] },
    Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 0.0, 0.0] },
];

#[rustfmt::skip]
const INDICES: &[u32] = &[
    // Front face
    0, 1, 2,
    2, 3, 0,

    // Right face
    1, 5, 6,
    6, 2, 1,

    // Back face
    5, 4, 7,
    7, 6, 5,

    // Left face
    4, 0, 3,
    3, 7, 4,

    // Top face
    3, 2, 6,
    6, 7, 3,

    // Bottom face
    4, 5, 1,
    1, 0, 4,
];

/*
#[rustfmt::skip]
const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5, 0.5, 0.0], color: [0.1, 0.1, 1.0] },
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.1, 1.0, 0.1] },
    Vertex { position: [0.5, -0.5, 0.0], color: [1.0, 0.1, 0.1] },
];

#[rustfmt::skip]
const INDICES: &[u32] = &[
    0, 1, 2
];*/

const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.003,
    g: 0.003,
    b: 0.005,
    a: 1.0,
};

fn mkbuffer<T>(device: &wgpu::Device, contents: &[T], usages: wgpu::BufferUsages) -> wgpu::Buffer
where
    T: bytemuck::Pod + bytemuck::Zeroable,
{
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(contents),
        usage: usages,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Show logs from wgpu
    env_logger::init();
    let mut keymap = keymap::Keymap::new();

    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window(NAME, 1200, 900)
        .resizable()
        .fullscreen()
        .build()
        .map_err(|e| e.to_string())?;
    let (width, height) = window.size();
    sdl_context.mouse().set_relative_mouse_mode(&window, true);
    sdl_context.mouse().capture(true);
    sdl_context.mouse().show_cursor(false);

    let instance = wgpu::Instance::new(&InstanceDescriptor::default());
    let surface = create_surface::create_surface(&instance, &window)?;
    let adapter_opt = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    }));
    let Some(adapter) = adapter_opt else {
        return Err("No adapter found".into());
    };

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::default(),
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))?;

    let capabilities = surface.get_capabilities(&adapter);
    let formats = capabilities.formats;
    let main_format = *formats
        .iter()
        .find(|format| format.is_srgb())
        .unwrap_or(&formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: main_format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        desired_maximum_frame_latency: 0,
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    let prepass_shader = device.create_shader_module(wgpu::include_wgsl!("prepass.wgsl"));
    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
    let screen_shader = device.create_shader_module(wgpu::include_wgsl!("screen.wgsl"));

    let vbuffer = mkbuffer(&device, VERTICES, wgpu::BufferUsages::VERTEX);
    let ibuffer = mkbuffer(&device, INDICES, wgpu::BufferUsages::INDEX);

    let mut depthtex = create_depth_texture(&device, "depth tex 1");

    let mut cam = Camera {
        eye: Vec3::Z * 3.,
        forw: Vec3::NEG_Z,
        up: Vec3::Y,
        aspect: width as f32 / height as f32,
        fovy: 90.0,
        znear: 0.1,
        zfar: 100.0,
        speed: 0.05,
        yaw: PI * 1.5,
        pitch: 0.0,
        sens: 0.1,
        pitch_max: 87.0,
    };

    let (qt_bgl, qt_bg) = mk_texbg(&device, &queue, include_bytes!("qt.png"), "quake texture");

    let mut uniforms = Uniforms::new();
    uniforms.update(&cam, Mat4::IDENTITY);

    let uniform_buffer = mkbuffer(
        &device,
        &[uniforms],
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );

    let uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: None,
        });

    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &uniform_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
        label: None,
    });

    let render_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render texture"),
        size: wgpu::Extent3d {
            width: SCR_SIZE.0,
            height: SCR_SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: main_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let render_tex_view = render_tex.create_view(&Default::default());

    let render_tex_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render texture bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

    let postprocess_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("post-process sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Post-process Texture Bind Group"),
        layout: &render_tex_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&postprocess_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&render_tex_view),
            },
        ],
    });

    let mut prepass_uniforms = PrepassUniforms { delta: 0.0 };

    let prepass_uniform_buffer = mkbuffer(
        &device,
        &[prepass_uniforms],
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );

    let prepass_uniform_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: None,
        });

    let prepass_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &prepass_uniform_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: prepass_uniform_buffer.as_entire_binding(),
        }],
        label: None,
    });

    let prepass_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&prepass_uniform_bind_group_layout],
        label: Some("prepass pipeline layout"),
        push_constant_ranges: &[],
    });

    let prepass_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: Some(&prepass_pipeline_layout),
        vertex: wgpu::VertexState {
            buffers: &[],
            module: &prepass_shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            targets: &[Some(wgpu::ColorTargetState {
                format: main_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            module: &prepass_shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        label: Some("main pipeline"),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&uniform_bind_group_layout, &qt_bgl],
        label: Some("texture pipeline layout"),
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            buffers: &[Vertex::desc()],
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            targets: &[Some(wgpu::ColorTargetState {
                format: main_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        label: Some("main pipeline"),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    let screen_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&render_tex_bind_group_layout],
        label: Some("screen pipeline layout"),
        push_constant_ranges: &[],
    });

    let screen_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: Some(&screen_pipeline_layout),
        vertex: wgpu::VertexState {
            buffers: &[],
            module: &screen_shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            targets: &[Some(wgpu::ColorTargetState {
                format: main_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            module: &screen_shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        label: Some("screen pipeline"),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    uniforms.update(&cam, Mat4::IDENTITY);

    let rot_val = 0.05;
    let background_move_val = 0.002;
    let cube_uv_move_val = 0.001;
    let mut model = Mat4::IDENTITY;
    let mut event_pump = sdl_context.event_pump()?;
    'running: loop {
        prepass_uniforms.delta += background_move_val;
        uniforms.delta += cube_uv_move_val;
        cam.update(&config, &keymap);
        model *= Mat4::from_rotation_x(rot_val) * Mat4::from_rotation_y(rot_val);
        uniforms.update(&cam, model);
        //uniforms.mvp = (Mat4::from_cols_array_2d(&uniforms.mvp) * model).to_cols_array_2d();
        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        queue.write_buffer(
            &prepass_uniform_buffer,
            0,
            bytemuck::cast_slice(&[prepass_uniforms]),
        );
        for event in event_pump.poll_iter() {
            match event {
                Event::Window {
                    window_id,
                    win_event:
                        WindowEvent::PixelSizeChanged(width, height)
                        | WindowEvent::Resized(width, height),
                    ..
                } if window_id == window.id() => {
                    config.width = width as u32;
                    config.height = height as u32;
                    depthtex = create_depth_texture(&device, "depth tex 1");
                    surface.configure(&device, &config);
                }
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    break 'running;
                }
                Event::KeyDown {
                    keycode: Some(Keycode::Tab),
                    ..
                } => {
                    // do not the keydown (maybe capture mouse here?)
                }
                Event::KeyDown { keycode, .. } => {
                    if let Some(kc) = keycode {
                        keymap.press(kc);
                    }
                }
                Event::KeyUp { keycode, .. } => {
                    if let Some(kc) = keycode {
                        keymap.release(kc);
                    }
                } 
                Event::MouseMotion { xrel, yrel, .. } => {
                    cam.look(xrel, yrel);
                }
                _ => (),
            }
        }

        let frame = match surface.get_current_texture() {
            Ok(frame) => frame,
            Err(err) => {
                let reason = match err {
                    SurfaceError::Timeout => "Timeout",
                    SurfaceError::Outdated => "Outdated",
                    SurfaceError::Lost => "Lost",
                    SurfaceError::OutOfMemory => "OutOfMemory",
                    SurfaceError::Other => "Other",
                };
                panic!("Failed to get current surface texture! Reason: {}", reason)
            }
        };

        let output = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("command_encoder"),
        });

        // prepass - draw background to texture
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_tex_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                label: Some("texture pass"),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // render()
            rpass.set_pipeline(&prepass_pipeline);
            rpass.set_bind_group(0, &prepass_uniform_bind_group, &[]);
            // shader!!!
            rpass.draw(0..6, 0..1);
        }

        // first pass - draw to texture
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_tex_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depthtex.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                label: Some("texture pass"),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // render()
            rpass.set_pipeline(&render_pipeline);
            rpass.set_bind_group(0, &uniform_bind_group, &[]);
            rpass.set_bind_group(1, &qt_bg, &[]);
            // NEW!
            rpass.set_vertex_buffer(0, vbuffer.slice(..));
            rpass.set_index_buffer(ibuffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..INDICES.len() as u32, 0, 0..1);
        }

        // second pass - draw to actual screen
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                label: Some("screen pass"),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&screen_pipeline);
            rpass.set_bind_group(0, &texture_bind_group, &[]);
            rpass.draw(0..6, 0..1);
        }
        queue.submit([encoder.finish()]);
        frame.present();
    }

    Ok(())
}

// surface creation functionality that links to the sdl window using a raw window handle
mod create_surface {
    use sdl3::video::Window;
    use wgpu::rwh::{HasDisplayHandle, HasWindowHandle};

    // contains the unsafe impl as much as possible by putting it in this module
    struct SyncWindow<'a>(&'a Window);

    unsafe impl<'a> Send for SyncWindow<'a> {}
    unsafe impl<'a> Sync for SyncWindow<'a> {}

    impl<'a> HasWindowHandle for SyncWindow<'a> {
        fn window_handle(&self) -> Result<wgpu::rwh::WindowHandle<'_>, wgpu::rwh::HandleError> {
            self.0.window_handle()
        }
    }
    impl<'a> HasDisplayHandle for SyncWindow<'a> {
        fn display_handle(&self) -> Result<wgpu::rwh::DisplayHandle<'_>, wgpu::rwh::HandleError> {
            self.0.display_handle()
        }
    }

    pub fn create_surface<'a>(
        instance: &wgpu::Instance,
        window: &'a Window,
    ) -> Result<wgpu::Surface<'a>, String> {
        instance
            .create_surface(SyncWindow(&window))
            .map_err(|err| err.to_string())
    }
}
