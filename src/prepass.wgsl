// screen space shader

const POSITIONS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
);

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Uniforms {
    delta: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(POSITIONS[vertex_index], 0.0, 1.0);
    out.uv = (out.pos.xy + vec2<f32>(1.0)) * 0.5;
    return out;
}

fn whitemix(d: f32) -> vec4<f32> {
    return vec4<f32>(mix(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.9, 0.7, 1.0), d).xyz, 1.0);
}

fn float_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn col(v: f32) -> vec4<f32> {
    return vec4<f32>(mix(vec3<f32>(0.3, 0.0, 0.6), vec3<f32>(1.0, 1.0, 1.0), v), 1.0);
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

fn ablend(c1: vec4<f32>, c2: vec4<f32>) -> vec4<f32> {
    return c1 * (1.0 - c2.a) + c2 * c2.a;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(in.uv.x + uniforms.delta, in.uv.y + uniforms.delta + hash(in.uv) * 20);
    let center = vec2<f32>(0.5, 0.5);
    let value = 1 - distance(in.uv, center);
    let scale = 40.0;
    return (whitemix(fbm(uv * scale)) - 0.5) * value;
}
