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

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(POSITIONS[vertex_index], 0.0, 1.0);
    out.uv = (out.pos.xy + vec2<f32>(1.0)) * 0.5;
    return out;
}

@group(0) @binding(0)
var uSampler: sampler;
@group(0) @binding(1)
var uTexture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let flipped_uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    let basecol = textureSample(uTexture, uSampler, flipped_uv);
    let val = 12.0;
    let nr = round(basecol.r * 255.0 / val) * val / 255.0;
    let ng = round(basecol.g * 255.0 / val) * val / 255.0;
    let nb = round(basecol.b * 255.0 / val) * val / 255.0;
    let dithered = vec4<f32>(nr, ng, nb, basecol.a);
    return dithered;
}
