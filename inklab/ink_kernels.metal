// Ink-bath fluid kernels of the game measured in the paper: classic stable
// fluids (Stam 1999) on a 128x256 grid, run as a compute prepass of about
// thirty dispatches per step. Copied verbatim from the game's shader source,
// which did not change between 2026-08-16 and the end of the measurements.
// Velocity is in texels per second; splats are gaussian dye drops and impulses.
#include <metal_stdlib>
using namespace metal;

struct InkSplatG {          // mirrors Swift's InkSplatGPU (3 x float4)
    float4 posVel;          // pos.xy (texels), impulse vel.zw (texels/s)
    float4 color;           // dye rgb, w unused
    float4 params;          // radius (texels), dyeStrength, radialVel, unused
};
struct InkU {               // mirrors Swift's InkUniforms (2 x float4)
    float4 a;               // dt, velDamp (/s), dyeDamp (/s), splatCount
    float4 b;               // vorticityEps, simW, simH, unused
};

constexpr sampler inkSamp(coord::normalized, filter::linear, address::clamp_to_edge);

// Advect velocity + dye semi-Lagrangian style, apply damping and splats.
kernel void ink_step(texture2d<float, access::sample> velIn  [[texture(0)]],
                     texture2d<float, access::sample> dyeIn  [[texture(1)]],
                     texture2d<float, access::write>  velOut [[texture(2)]],
                     texture2d<float, access::write>  dyeOut [[texture(3)]],
                     constant InkU& u [[buffer(0)]],
                     constant InkSplatG* splats [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    int W = int(u.b.y), H = int(u.b.z);
    if (int(gid.x) >= W || int(gid.y) >= H) return;
    float2 sim = float2(u.b.y, u.b.z);
    float2 p = float2(gid) + 0.5;
    float dt = u.a.x;
    float2 v = velIn.sample(inkSamp, p / sim).xy;
    float2 back = (p - v * dt) / sim;
    float2 nv = velIn.sample(inkSamp, back).xy * exp(-dt * u.a.y);
    float3 nd = dyeIn.sample(inkSamp, back).rgb * exp(-dt * u.a.z);
    for (int i = 0; i < int(u.a.w); i++) {
        InkSplatG s = splats[i];
        float2 d = p - s.posVel.xy;
        float r = max(s.params.x, 0.5);
        float g = exp(-dot(d, d) / (r * r));
        nv += s.posVel.zw * g;
        float dl = length(d);
        if (dl > 0.001) nv += (d / dl) * (s.params.z * g);
        nd += s.color.rgb * (s.params.y * g);
    }
    // Closed box: border texels are walls.
    if (gid.x == 0 || gid.y == 0 || int(gid.x) == W - 1 || int(gid.y) == H - 1) nv = 0.0;
    velOut.write(float4(nv, 0, 0), gid);
    // Clamp above 1 so heavy overlap whitens instead of running away.
    dyeOut.write(float4(min(nd, 1.6), 1), gid);
}

static inline float ink_curl(texture2d<float, access::read> vel, int2 p, int W, int H) {
    uint2 l = uint2(max(p.x - 1, 0), p.y),      r = uint2(min(p.x + 1, W - 1), p.y);
    uint2 up = uint2(p.x, max(p.y - 1, 0)),     dn = uint2(p.x, min(p.y + 1, H - 1));
    return 0.5 * ((vel.read(r).y - vel.read(l).y) - (vel.read(dn).x - vel.read(up).x));
}

// Vorticity confinement: push velocity around local curl maxima so the ink
// keeps swirling instead of diffusing into laminar mush.
kernel void ink_vorticity(texture2d<float, access::read>  velIn  [[texture(0)]],
                          texture2d<float, access::write> velOut [[texture(1)]],
                          constant InkU& u [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    int W = int(u.b.y), H = int(u.b.z);
    if (int(gid.x) >= W || int(gid.y) >= H) return;
    int2 p = int2(gid);
    float cC = ink_curl(velIn, p, W, H);
    float cL = ink_curl(velIn, int2(max(p.x - 1, 0), p.y), W, H);
    float cR = ink_curl(velIn, int2(min(p.x + 1, W - 1), p.y), W, H);
    float cU = ink_curl(velIn, int2(p.x, max(p.y - 1, 0)), W, H);
    float cD = ink_curl(velIn, int2(p.x, min(p.y + 1, H - 1)), W, H);
    float2 grad = 0.5 * float2(abs(cR) - abs(cL), abs(cD) - abs(cU));
    float2 nv = velIn.read(gid).xy;
    float len = length(grad);
    if (len > 1e-5) {
        float2 n = grad / len;
        nv += u.b.x * cC * float2(n.y, -n.x) * u.a.x;
    }
    velOut.write(float4(nv, 0, 0), gid);
}

kernel void ink_divergence(texture2d<float, access::read>  vel [[texture(0)]],
                           texture2d<float, access::write> div [[texture(1)]],
                           constant InkU& u [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    int W = int(u.b.y), H = int(u.b.z);
    if (int(gid.x) >= W || int(gid.y) >= H) return;
    int2 p = int2(gid);
    float L = vel.read(uint2(max(p.x - 1, 0), p.y)).x;
    float R = vel.read(uint2(min(p.x + 1, W - 1), p.y)).x;
    float U = vel.read(uint2(p.x, max(p.y - 1, 0))).y;
    float D = vel.read(uint2(p.x, min(p.y + 1, H - 1))).y;
    div.write(float4(0.5 * ((R - L) + (D - U)), 0, 0, 0), gid);
}

kernel void ink_jacobi(texture2d<float, access::read>  prIn  [[texture(0)]],
                       texture2d<float, access::read>  div   [[texture(1)]],
                       texture2d<float, access::write> prOut [[texture(2)]],
                       constant InkU& u [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    int W = int(u.b.y), H = int(u.b.z);
    if (int(gid.x) >= W || int(gid.y) >= H) return;
    int2 p = int2(gid);
    float L = prIn.read(uint2(max(p.x - 1, 0), p.y)).x;
    float R = prIn.read(uint2(min(p.x + 1, W - 1), p.y)).x;
    float U = prIn.read(uint2(p.x, max(p.y - 1, 0))).x;
    float D = prIn.read(uint2(p.x, min(p.y + 1, H - 1))).x;
    prOut.write(float4((L + R + U + D - div.read(gid).x) * 0.25, 0, 0, 0), gid);
}

kernel void ink_project(texture2d<float, access::read>  velIn  [[texture(0)]],
                        texture2d<float, access::read>  pr     [[texture(1)]],
                        texture2d<float, access::write> velOut [[texture(2)]],
                        constant InkU& u [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    int W = int(u.b.y), H = int(u.b.z);
    if (int(gid.x) >= W || int(gid.y) >= H) return;
    int2 p = int2(gid);
    float L = pr.read(uint2(max(p.x - 1, 0), p.y)).x;
    float R = pr.read(uint2(min(p.x + 1, W - 1), p.y)).x;
    float U = pr.read(uint2(p.x, max(p.y - 1, 0))).x;
    float D = pr.read(uint2(p.x, min(p.y + 1, H - 1))).x;
    float2 nv = velIn.read(gid).xy - 0.5 * float2(R - L, D - U);
    if (gid.x == 0 || gid.y == 0 || int(gid.x) == W - 1 || int(gid.y) == H - 1) nv = 0.0;
    velOut.write(float4(nv, 0, 0), gid);
}

kernel void ink_zero(texture2d<float, access::write> tex [[texture(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
    tex.write(float4(0), gid);
}
