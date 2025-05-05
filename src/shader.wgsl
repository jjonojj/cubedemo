// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct Uniforms {
    mvp: mat4x4<f32>,
    delta: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = uniforms.mvp * vec4<f32>(model.position, 1.0);
    return out;
}

fn col(v: f32) -> vec4<f32> {
    return vec4<f32>(mix(vec3<f32>(0.15, 0.0, 0.6), vec3<f32>(1.0, 1.0, 1.0), v), 1.0);
}

fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(hash(i + vec2<f32>(0.0, 0.0)), hash(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

fn fbm(p: vec2<f32>) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.4;
    var shift = vec2<f32>(100.0, 100.0);
    var freq_p = p;

    for (var i = 0; i < 5; i = i + 1) {
        value = value + amplitude * noise(freq_p);
        freq_p = freq_p * 2.0 + shift;
        amplitude = amplitude * 0.5;
    }

    return value;
}

// Fragment shader

fn gray(v: f32) -> vec4<f32> {
    return vec4<f32>(v, v, v, 1.0);
}

@fragment
fn fs_main(
    @location(0) color: vec3<f32>,
    @builtin(position) frag_coord: vec4<f32>
) -> @location(0) vec4<f32> {
    // return vec4<f32>(color, 1.0);
    // return textureSample(t_diffuse, s_diffuse, uv);

    let scale = 40.0;
    let resolution = vec2<f32>(360, 270);
    let uv = frag_coord.xy / resolution;
    let fuv = vec2<f32>(uv.x + uniforms.delta + sin(uniforms.delta * 50.0 + uv.y * 20.0) * 0.03, uv.y + uniforms.delta + sin(uniforms.delta * 50.0 + uv.x * 20.0) * 0.03);
    return col(fbm(fuv * scale)) - 0.3;
}
