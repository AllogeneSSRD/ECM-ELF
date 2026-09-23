#!/usr/bin/env python3

"""Shared helpers for AMDGCN fused add/sub-mod asm block generators."""



from __future__ import annotations



from pathlib import Path





def emit_c_fix_add(n: int) -> str:

    lines = [f"static inline void c_fix_add_n{n}(\n"]

    for i in range(n):

        lines.append(f"    uint *r{i}, ")

    lines[-1] = lines[-1].rstrip(", ") + ",\n"

    for i in range(n):

        comma = "," if i < n - 1 else ") {\n"

        lines.append(f"    uint n{i}{comma}")

    lines.append("    ulong c = 0ul;\n    ulong s;\n")

    for i in range(n):

        lines.append(

            f"    s = (ulong)*r{i} + (ulong)n{i} + c; c = s >> 32; *r{i} = (uint)s;\n"

        )

    lines.append("}\n\n")

    return "".join(lines)





def _scalar_loads(prefix: str, src: str, n: int) -> str:

    return "".join(f"    uint {prefix}{i} = {src}[{i}];\n" for i in range(n))





def _ptr_qual(global_addr: bool) -> str:

    return "__global " if global_addr else ""





def emit_add_block(n: int, global_addr: bool = True, fix_in_block: bool = True) -> str:

    p = _ptr_qual(global_addr)

    suffix = "" if global_addr else "_priv"
    if global_addr is False and fix_in_block is False:
        suffix = "_priv_nofix"

    loads_a = _scalar_loads("a", "a", n)

    loads_b = _scalar_loads("b", "b", n)

    loads_n = _scalar_loads("n", "n", n)

    decl_s = ", ".join(f"s{i}" for i in range(n))

    decl_r = ", ".join(f"r{i}" for i in range(n))

    decl_nn = ", ".join(f"n{i}n" for i in range(n))



    asm_add = ['        "v_cmp_eq_u32    vcc_lo, %[ca_bit], %[o]\\n\\t"']

    for i in range(n):

        asm_add.append(

            f'        "v_add_co_ci_u32 %[s{i}], vcc_lo, %[a{i}], %[b{i}], vcc_lo\\n\\t"'

        )

    asm_add.append('        "v_cndmask_b32   %[ca], %[z], %[o], vcc_lo\\n\\t"')

    for i in range(n):

        asm_add.append(f'        "v_not_b32       %[n{i}n], %[n{i}]\\n\\t"')

    asm_add.append('        "v_cmp_eq_u32    vcc_lo, %[cs_bit], %[o]\\n\\t"')

    for i in range(n - 1):

        asm_add.append(

            f'        "v_add_co_ci_u32 %[r{i}], vcc_lo, %[s{i}], %[n{i}n], vcc_lo\\n\\t"'

        )

    asm_add.append(

        f'        "v_add_co_ci_u32 %[r{n - 1}], vcc_lo, %[s{n - 1}], %[n{n - 1}n], vcc_lo\\n\\t"'

    )

    asm_add.append('        "v_cndmask_b32   %[cs], %[z], %[o], vcc_lo"')



    out_s = ", ".join(f'[s{i}] "=&v"(s{i})' for i in range(n))

    out_r = ", ".join(f'[r{i}] "=&v"(r{i})' for i in range(n))

    out_nn = ", ".join(f'[n{i}n] "=&v"(n{i}n)' for i in range(n))

    in_ab = ", ".join(f'[a{i}] "v"(a{i}), [b{i}] "v"(b{i})' for i in range(n))

    in_n = ", ".join(f'[n{i}] "v"(n{i})' for i in range(n))

    fix_call = ", ".join(f"&r{i}" for i in range(n))

    fix_n = ", ".join(f"n{i}" for i in range(n))

    stores = "".join(f"    r[{i}] = r{i};\n" for i in range(n))

    tag = "global bench" if global_addr else "private stage1"

    fix_section = ""

    if fix_in_block:

        fix_section = f"""

    if ((ca | cs) == 0u) {{

        c_fix_add_n{n}({fix_call}, {fix_n});

    }}

"""



    return f"""

// {n}-limb fused add-mod block ({tag}).

static inline void asm_fused_block{n}{suffix}({p}const uint *a, {p}const uint *b,

                                   {p}const uint *n, {p}uint *r, uint ca_in,

                                   uint cs_in, uint *ca_out, uint *cs_out) {{

{loads_a}{loads_b}{loads_n}    uint {decl_s};

    uint {decl_r};

    uint {decl_nn};

    uint ca = 0u, cs = 0u;

    const uint z = 0u, o = 1u;

    uint ca_bit = ca_in ? o : z;

    uint cs_bit = cs_in ? o : z;



    __asm volatile(

{chr(10).join(asm_add)}

        : {out_s},

          {out_r},

          {out_nn},

          [ca] "=&v"(ca), [cs] "=&v"(cs)

        : {in_ab},

          {in_n},

          [ca_bit] "v"(ca_bit), [cs_bit] "v"(cs_bit), [z] "v"(z), [o] "v"(o)

        : "vcc_lo");

{fix_section}

{stores}    *ca_out = ca;

    *cs_out = cs;

}}

"""





def emit_sub_block(n: int, global_addr: bool = True, fix_in_block: bool = True) -> str:

    p = _ptr_qual(global_addr)

    suffix = "" if global_addr else "_priv"
    if global_addr is False and fix_in_block is False:
        suffix = "_priv_nofix"

    loads_a = _scalar_loads("a", "a", n)

    loads_b = _scalar_loads("b", "b", n)

    loads_n = _scalar_loads("n", "n", n)

    decl_r = ", ".join(f"r{i}" for i in range(n))



    asm_sub = ['        "v_cmp_eq_u32    vcc_lo, %[br_bit], %[o]\\n\\t"']

    for i in range(n):

        asm_sub.append(

            f'        "v_sub_co_ci_u32 %[r{i}], vcc_lo, %[a{i}], %[b{i}], vcc_lo\\n\\t"'

        )

    asm_sub.append('        "v_cndmask_b32   %[br], %[z], %[o], vcc_lo"')



    out_r = ", ".join(f'[r{i}] "=&v"(r{i})' for i in range(n))

    in_ab = ", ".join(f'[a{i}] "v"(a{i}), [b{i}] "v"(b{i})' for i in range(n))

    fix_call = ", ".join(f"&r{i}" for i in range(n))

    fix_n = ", ".join(f"n{i}" for i in range(n))

    stores = "".join(f"    r[{i}] = r{i};\n" for i in range(n))

    tag = "global bench" if global_addr else "private stage1"

    fix_section = ""

    if fix_in_block:

        fix_section = f"""

    if (br != 0u) {{

        c_fix_add_n{n}({fix_call}, {fix_n});

    }}

"""



    return f"""

// {n}-limb fused sub-mod block ({tag}).

static inline void asm_sub_fused_block{n}{suffix}({p}const uint *a, {p}const uint *b,

                                        {p}const uint *n, {p}uint *r, uint br_in,

                                        uint *br_out) {{

{loads_a}{loads_b}{loads_n}    uint {decl_r};

    uint br = 0u;

    const uint z = 0u, o = 1u;

    uint br_bit = br_in ? o : z;



    __asm volatile(

{chr(10).join(asm_sub)}

        : {out_r},

          [br] "=&v"(br)

        : {in_ab},

          [br_bit] "v"(br_bit), [z] "v"(z), [o] "v"(o)

        : "vcc_lo");

{fix_section}

{stores}    *br_out = br;

}}

"""





def write_add_block_file(out: Path, n: int, generator_name: str) -> None:

    parts = [

        f"// AUTO-GENERATED by tools/{generator_name} — do not edit.\n",

        "#if defined(MP_ADDMOD_ASM_ENABLE) && defined(__AMDGCN__)\n\n",

        emit_c_fix_add(n),

        emit_add_block(n, global_addr=True),

        "#endif\n",

    ]

    out.write_text("".join(parts), encoding="utf-8")





def write_sub_block_file(out: Path, n: int, generator_name: str) -> None:

    parts = [

        f"// AUTO-GENERATED by tools/{generator_name} — do not edit.\n",

        "#if defined(MP_ADDMOD_ASM_ENABLE) && defined(__AMDGCN__)\n\n",

        emit_sub_block(n, global_addr=True),

        "#endif\n",

    ]

    out.write_text("".join(parts), encoding="utf-8")

