// F/G tree construction, layout, degree and root selection.
#include "tree.h"
#include "options.h"

std::vector<std::vector<unsigned long long>> build_tree_flat(
    PolyLayer &L,
    const std::vector<std::vector<unsigned long long>> &leaf,
    std::vector<size_t> &deg,
    size_t &pad_out,
    FTreeStats &fs,
    int cat,
    bool keep_children)
{
    stage2_trace::Range trace("Tree.host_driver");
    const size_t W = L.W;
    const size_t n = leaf.size();
    size_t pad = 1;
    while (pad < n) pad *= 2;
    fs.leaves = n;
    fs.padded = pad;
    pad_out = pad;
    /* Root-only callers never read a consumed child again. Internal placeholders need no
       payload; leaves, including constant-one padding, are defined before the first level. */
    std::vector<std::vector<unsigned long long>> t = keep_children
        ? std::vector<std::vector<unsigned long long>>(2 * pad, std::vector<unsigned long long>(W, 0ull))
        : std::vector<std::vector<unsigned long long>>(2 * pad);
    deg.assign(2 * pad, 0);
    for (size_t i = keep_children ? 0 : pad; i < 2 * pad; ++i) {
        if (!keep_children) t[i].assign(W, 0ull);
        t[i][0] = 1;                                        /* the constant 1 */
    }
    for (size_t i = 0; i < n; ++i) { t[pad + i] = leaf[i]; deg[pad + i] = 1; }
    size_t live_words = 0;
    for (const auto &v : t) live_words += v.capacity();
    fs.node_peak_bytes = 8ull * live_words;
    auto update_parent = [&](size_t i, size_t old_cap) {
        live_words -= old_cap; live_words += t[i].capacity();
        fs.node_peak_bytes = std::max(fs.node_peak_bytes, 8ull * live_words);
    };
    auto passthrough = [&](size_t i, size_t child) {
        const size_t old = t[i].capacity(), source = t[child].capacity();
        if (keep_children) { t[i] = t[child]; update_parent(i, old); }
        else {
            t[i] = std::move(t[child]);
            live_words -= old + source;
            live_words += t[i].capacity() + t[child].capacity();
            fs.node_peak_bytes = std::max(fs.node_peak_bytes, 8ull * live_words);
            ++fs.passthrough_moves;
        }
        deg[i] = deg[child];
    };
    auto release_children = [&](size_t first, size_t last) {
        if (keep_children) return;
        const double tr0 = now_s();
        for (size_t i = first; i < last; ++i) {
            const size_t old = t[i].capacity();
            std::vector<unsigned long long>().swap(t[i]);
            live_words -= old; fs.node_released_bytes += 8ull * old;
            ++fs.nodes_released;
        }
        fs.t_release += now_s() - tr0;
    };
    auto finish = [&] {
        fs.node_retained_bytes = 8ull * live_words;
        if (g_s4_carry_trace) {
            size_t actual = 0;
            for (size_t i=0; i<t.size(); ++i) {
                actual += t[i].capacity();
                if (!keep_children && i>1 && !t[i].empty()) {
                    std::fprintf(stderr, "%s: FATAL: root-only tree retains child %llu\n", NTT_PROBE_NAME,
                                 (unsigned long long)i); std::exit(3);
                }
            }
            if (actual != live_words) {
                std::fprintf(stderr, "%s: FATAL: tree capacity accounting mismatch\n", NTT_PROBE_NAME);
                std::exit(3);
            }
        }
    };
    /* A one-leaf tree has no parent level (base=0 would never reach the loop's base=1 stop). */
    if (pad == 1) { finish(); return t; }
    if (!L.s4) {                                               /* the pre-S4 path, unchanged */
        for (size_t i = pad; i-- > 1; ) {
            if (deg[2*i] == 0 && poly_is_one(t[2 * i].data(), W)) { passthrough(i, 2*i+1); }
            else if (deg[2*i+1] == 0 && poly_is_one(t[2 * i + 1].data(), W)) { passthrough(i, 2*i); }
            else {
                const size_t old = t[i].capacity();
                t[i] = poly_mul_modN(L, t[2 * i], deg[2 * i], t[2 * i + 1], deg[2 * i + 1], cat);
                update_parent(i, old);
                deg[i] = deg[2 * i] + deg[2 * i + 1];
                ++fs.muls;
            }
            release_children(2*i, 2*i+2);
        }
        finish();
        return t;
    }
    ++L.s4->level_calls;
    /* ---- PER-LEVEL TIMES OF A TREE (section 26) ---------------------------------------------
       The cost model says every level of a product tree carries the SAME operand bits (P*slot_bits
       per level, whatever the degree), and the measured per-phase rates (g_tree 0.32 ns per operand
       bit, fold 0.38, descent 0.33) say the phases are uniformly efficient -- which would make the
       G tree's 101 s inherent rather than wasteful.  That is a big claim to leave on arithmetic, so
       each level is timed and the first two builds (the F tree, then the first G tree) print their
       ladder: if the levels are equal, the tree has no cheap half to attack. */
    static int lvl_print = 0;
    const bool lvl_trace = (lvl_print < 2);
    if (lvl_trace) ++lvl_print;
    for (size_t base = pad / 2; ; base /= 2) {
        const double tl0 = now_s();
        unsigned long long lvl_muls = 0, lvl_groups = 0;
        double lvl_fwd = 0, lvl_inv = 0, lvl_slot = 0, lvl_hpack = 0, lvl_h2d = 0, lvl_check = 0;
        double lvl_plan = 0, lvl_opcopy = 0, lvl_hout = 0, lvl_total = 0;
        /* nodes [base, 2*base), children at [2*base, 4*base) -- all children are already built */
        std::map<std::pair<size_t, size_t>, std::vector<size_t>> groups;
        for (size_t i = base; i < 2 * base; ++i) {
            const size_t c0 = 2 * i, c1 = 2 * i + 1;
            if (deg[c0] == 0 && poly_is_one(t[c0].data(), W)) { passthrough(i, c1); }
            else if (deg[c1] == 0 && poly_is_one(t[c1].data(), W)) { passthrough(i, c0); }
            else {
                const size_t ma = deg[c0] + 1, mb = deg[c1] + 1;
                groups[std::make_pair((ma < mb) ? ma : mb, (ma < mb) ? mb : ma)].push_back(i);
            }
        }
        for (auto &g : groups) {
            const size_t ma = g.first.first, mb = g.first.second;
            const size_t P = (ma > mb) ? ma : mb;
            const size_t nbatch = g.second.size();
            const size_t nc = ma + mb - 1;
            std::vector<unsigned long long> wa(nbatch * P * W, 0ull), wb(nbatch * P * W, 0ull);
            for (size_t s = 0; s < nbatch; ++s) {
                const size_t i = g.second[s];
                /* the operand lengths are the CHILDREN's own (the group key is normalised to
                   (min,max) for the shape, so it must NOT be used to size the copies) */
                const size_t na = deg[2 * i] + 1, nb2 = deg[2 * i + 1] + 1;
                std::copy(t[2 * i].begin(), t[2 * i].begin() + (long)(na * W),
                          wa.begin() + (long)(s * P * W));
                std::copy(t[2 * i + 1].begin(), t[2 * i + 1].begin() + (long)(nb2 * W),
                          wb.begin() + (long)(s * P * W));
            }
            std::vector<unsigned long long> res;
            NttMulStats gst{};
            poly_mul_batch_modN(L, wa.data(), wb.data(), ma, mb, nbatch, res, cat, &gst);
            ++L.s4->groups;
            ++lvl_groups;
            lvl_muls += nbatch;
            if (lvl_trace) {
                lvl_fwd += gst.t_fwd; lvl_inv += gst.t_inv; lvl_slot += gst.t_slot;
                lvl_hpack += gst.t_hpack; lvl_h2d += gst.t_h2d_batch;
                lvl_check += gst.t_check; lvl_plan += gst.t_plan;
                lvl_opcopy += gst.t_opcopy; lvl_hout += gst.t_hout;
                lvl_total += gst.t_total_call;
            }
            for (size_t s = 0; s < nbatch; ++s) {
                const size_t i = g.second[s];
                const size_t old = t[i].capacity();
                t[i].assign(res.begin() + (long)(s * nc * W), res.begin() + (long)((s + 1) * nc * W));
                update_parent(i, old);
                deg[i] = ma + mb - 2;
                ++fs.muls;
            }
        }
        /* Inputs were copied into wa/wb before the call; all device readers use that owned
           staging. Every sibling group has finished consuming this level before reclamation. */
        release_children(2*base, 4*base);
        if (lvl_trace)
            stage2_log::print(stage2_log::debug, "tree_level: base=%llu groups=%llu muls=%llu t=%.3f s | ntt total=%.3f "
                        "fwd=%.3f inv=%.3f slot=%.3f hpack=%.3f h2d=%.3f check=%.3f plan=%.3f "
                        "opcopy=%.3f hout=%.3f\n",
                        (unsigned long long)base, lvl_groups, lvl_muls, now_s() - tl0, lvl_total,
                        lvl_fwd, lvl_inv, lvl_slot, lvl_hpack, lvl_h2d, lvl_check, lvl_plan,
                        lvl_opcopy, lvl_hout);
        if (base == 1) break;
    }
    finish();
    return t;
}

std::vector<unsigned long long> groot_product_gmp(
    PolyLayer &L,
    const std::vector<unsigned long long> &a,
    const std::vector<unsigned long long> &b)
{
    const size_t W=L.W,ma=a.size()/W,mb=b.size()/W;
    std::vector<unsigned long long> out((ma+mb-1)*W,0),word(W);
    mpz_t x,y,z;mpz_inits(x,y,z,nullptr);
    for(size_t k=0;k<ma+mb-1;++k) {
        mpz_set_ui(z,0);
        for(size_t i=0;i<ma;++i) if(k>=i && k-i<mb) {
            words_to_mpz(x,a.data()+i*W,W);words_to_mpz(y,b.data()+(k-i)*W,W);mpz_addmul(z,x,y);
        }
        mpz_mod(z,z,L.N);mpz_to_words(word,W,z);std::copy(word.begin(),word.end(),out.begin()+k*W);
    }
    mpz_clears(x,y,z,nullptr);return out;
}

std::vector<std::vector<unsigned long long>> build_groot_device(
    PolyLayer &L,
    const std::vector<std::vector<unsigned long long>> &leaf,
    std::vector<size_t> &deg,
    size_t &pad_out,
    FTreeStats &fs,
    int cat,
    size_t device_n,
    const std::function<void(unsigned long long*)> &fill,
    const std::function<void(const unsigned long long*,size_t)> &root_sink,
    bool balanced)
{
    stage2_trace::Range trace("GTree.build");
    const size_t W=L.W,n=fill?device_n:leaf.size();
    ecm_stage2::GiantTreeLayout layout;
    if(!ecm_stage2::giant_tree_layout(n,layout)){std::fprintf(stderr,"FATAL: G tree layout overflow\n");std::exit(3);}
    const size_t pad=layout.pad;
    const bool check=g_groot_device_check;
    if(check && n>512) {std::fprintf(stderr,"%s: FATAL: resident GMP check limited to 512 leaves\n",NTT_PROBE_NAME);std::exit(3);}
    ++g_gdevice.trees;fs.leaves=n;fs.padded=pad;pad_out=pad;
    deg.assign(2*pad,0);
    for(size_t i=0;i<n;++i) {
        if(!fill && leaf[i].size()!=2*W) {std::fprintf(stderr,"%s: FATAL: resident tree requires linear leaves\n",NTT_PROBE_NAME);std::exit(3);}
        deg[pad+(balanced?layout.leaf(i):i)]=1;
    }
    for(size_t i=pad;--i;)deg[i]=deg[2*i]+deg[2*i+1];
    std::vector<std::vector<unsigned long long>> tree(2*pad),expected;
    if(!n) {tree[1].assign(W,0);tree[1][0]=1;fs.node_peak_bytes=fs.node_retained_bytes=8*W;return tree;}
    S4Ctx &C=*L.s4;
    // Exclusive lease of the existing raw staging pair. Resident calls never upload raw inputs.
    struct LeaseAudit {
        S4Ctx &ctx;size_t leaves;bool enabled;
        ~LeaseAudit(){if(enabled)ecm_stage2::s4_signature_boundary(ctx.s4_program_signature,ecm_stage2::S4TreeEnd,leaves);}
    } lease_audit{C,n,fuse_env_ull("NTT_REQUEST_AUDIT",0)!=0};
    if(lease_audit.enabled)ecm_stage2::s4_signature_boundary(C.s4_program_signature,ecm_stage2::S4TreeBegin,n);
    const size_t need=2*n*W;
    size_t next_need=0;
    for(size_t i=pad/2;pad>1 && i<pad;++i)if(deg[i])next_need+=(deg[i]+1)*W;
    // Dense linear leaves: frontier words = (n + nonempty nodes)*W, decreasing each level.
    // A holds the largest (leaf) frontier; B only needs the first parent frontier.
    C.raw_reserve(need,g_groot_compact_raw?next_need:need);
    g_gdevice.raw_peak_bytes=std::max(g_gdevice.raw_peak_bytes,8ull*(C.d_rawA_cap+C.d_rawB_cap));
    g_gmemory.rawA_peak_bytes=std::max(g_gmemory.rawA_peak_bytes,8ull*C.d_rawA_cap);
    g_gmemory.rawB_peak_bytes=std::max(g_gmemory.rawB_peak_bytes,8ull*C.d_rawB_cap);
    const double upload=now_s();
    bool available[2]={false,false},used[2]={false,false};
    if(g_groot_leaf_staging && g_s4_async)for(int k=0;k<2;++k)
        available[k]=g_pin_out[k] && g_pin_out_cap[k]>=2*W && g_pin_ev[k];
    if(fill)fill(C.d_rawA);
    else if(available[0] || available[1]) {
        ++g_gmemory.pinned_trees;size_t offset=0,turn=0;
        while(offset<n) {
            size_t k=turn++&1;if(!available[k])k^=1;
            // The last D2H or our previous H2D must consume this pinned slot before CPU writes it.
            CK(cudaEventSynchronize(g_pin_ev[k]));
            size_t take=std::min(n-offset,g_pin_out_cap[k]/(2*W));
            if(g_groot_leaf_chunk)take=std::min(take,g_groot_leaf_chunk);
            for(size_t i=0;i<take;++i)std::copy(leaf[offset+i].begin(),leaf[offset+i].end(),g_pin_out[k]+2*i*W);
            CK(cudaMemcpyAsync(C.d_rawA+2*offset*W,g_pin_out[k],take*2*W*8,cudaMemcpyHostToDevice));
            CK(cudaEventRecord(g_pin_ev[k]));used[k]=true;
            ++g_gmemory.pinned_slices;g_gmemory.pinned_words+=take*2*W;
            g_gmemory.pinned_borrow_peak_bytes=std::max(g_gmemory.pinned_borrow_peak_bytes,16ull*take*W);
            offset+=take;
        }
        // Trace multiplies may grow/free output staging at entry: finish EVERY leaf H2D first.
        // n=1 has no multiply, so it also needs this explicit endpoint fence.
        for(int k=0;k<2;++k)if(used[k])CK(cudaEventSynchronize(g_pin_ev[k]));
    } else {
        if(g_groot_leaf_staging)++g_gmemory.leaf_fallbacks;else ++g_gmemory.legacy_trees;
        std::vector<unsigned long long> initial(need);
        for(size_t i=0;i<n;++i)std::copy(leaf[i].begin(),leaf[i].end(),initial.begin()+2*i*W);
        g_gdevice.host_staging_peak_bytes=std::max(g_gdevice.host_staging_peak_bytes,8ull*initial.capacity());
        CK(cudaMemcpy(C.d_rawA,initial.data(),need*8,cudaMemcpyHostToDevice));g_gmemory.pageable_words+=need;
    }
    if(!fill){C.t_h2d_raw+=now_s()-upload;C.raw_words+=need;}
    g_gdevice.leaf_words+=need;
    if(check) {
        std::vector<unsigned long long> device_leaf;
        if(fill){device_leaf.resize(need);CK(cudaMemcpy(device_leaf.data(),C.d_rawA,need*8,cudaMemcpyDeviceToHost));}
        expected.resize(2*pad);
        for(size_t i=0;i<pad;++i){expected[pad+i].assign(W,0);expected[pad+i][0]=1;}
        for(size_t i=0;i<n;++i) {
            auto &dest=expected[pad+(balanced?layout.leaf(i):i)];
            if(fill)dest.assign(device_leaf.begin()+2*i*W,device_leaf.begin()+2*(i+1)*W);
            else dest=leaf[i];
        }
        for(size_t i=pad;--i;)expected[i]=groot_product_gmp(L,expected[2*i],expected[2*i+1]);
    }
    auto *cur=C.d_rawA,*next=C.d_rawB;size_t cur_words=need;
    std::vector<unsigned long long> offsets(pad);
    size_t leaf_offset=0;
    for(size_t i=0;i<pad;++i){offsets[i]=leaf_offset;leaf_offset+=2*W*deg[pad+i];}
    if(leaf_offset!=need){std::fprintf(stderr,"FATAL: G tree dense leaf offsets\n");std::exit(3);}
    struct Metadata {
        unsigned long long *device=nullptr,*pinned=nullptr;size_t capacity=0;
        ~Metadata(){if(device)CK(cudaFree(device));if(pinned)CK(cudaFreeHost(pinned));}
    } meta;
    if(pad>1) {
        meta.capacity=3*(pad/2);CK(cudaMalloc(&meta.device,meta.capacity*8));
        if(g_s4_async) {
            cudaError_t err=cudaHostAlloc(&meta.pinned,meta.capacity*8,cudaHostAllocDefault);
            if(err!=cudaSuccess){meta.pinned=nullptr;cudaGetLastError();}
        }
        g_gdevice.metadata_peak_bytes=std::max(g_gdevice.metadata_peak_bytes,8ull*meta.capacity);
    }
    auto validate=[&](size_t base) {
        if(!check)return;
        std::vector<unsigned long long> words(cur_words);
        CK(cudaMemcpy(words.data(),cur,cur_words*8,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<base;++i) {
            const size_t d=deg[base+i];++g_gdevice.checked_nodes;g_gdevice.checked_words+=(d+1)*W;
            if(d && !std::equal(expected[base+i].begin(),expected[base+i].end(),words.begin()+offsets[i])) {
                std::fprintf(stderr,"%s: FATAL: resident GMP node mismatch node=%llu degree=%llu\n",NTT_PROBE_NAME,
                    (unsigned long long)(base+i),(unsigned long long)d);std::exit(3);
            }
        }
    };
    validate(pad);
    const double start=now_s();
    static bool poisoned=false;
    for(size_t base=pad/2; base; base/=2) {
        stage2_trace::Range level_trace("GTree.level");
        const double level_start=now_s();
        std::vector<unsigned long long> target(base);size_t next_words=0;
        for(size_t i=0;i<base;++i) {target[i]=next_words;if(deg[base+i])next_words+=(deg[base+i]+1)*W;}
        if(next_words>C.raw_capacity(next)) {std::fprintf(stderr,"%s: FATAL: resident next frontier exceeds raw lease\n",NTT_PROBE_NAME);std::exit(3);}
        g_gdevice.logical_frontier_peak_bytes=std::max(g_gdevice.logical_frontier_peak_bytes,8ull*(cur_words+next_words));
        std::map<std::pair<size_t,size_t>,std::vector<size_t>> groups;
        for(size_t i=0;i<base;++i) {
            const size_t da=deg[2*base+2*i],db=deg[2*base+2*i+1];
            if(!da && !db)continue;
            if(!da || !db) {
                const size_t child=2*i+(da?0:1),d=da?da:db;
                CK(cudaMemcpyAsync(next+target[i],cur+offsets[child],(d+1)*W*8,cudaMemcpyDeviceToDevice));
                ++g_gdevice.copies;continue;
            }
            groups[{std::min(da,db)+1,std::max(da,db)+1}].push_back(i);
        }
        for(const auto &group:groups) {
            const size_t ma=group.first.first,mb=group.first.second,nb=group.second.size();
            std::vector<unsigned long long> map(3*nb);
            for(size_t j=0;j<nb;++j) {
                const size_t i=group.second[j],left=2*i,right=left+1;
                const bool swap=deg[2*base+left]>deg[2*base+right];
                map[j]=offsets[swap?right:left];map[nb+j]=offsets[swap?left:right];map[2*nb+j]=target[i];
            }
            // Previous metadata upload is complete when the multiply's final carry check returns.
            // All scatter reads of DEVICE metadata precede this update in the default stream.
            if(meta.pinned) {
                std::copy(map.begin(),map.end(),meta.pinned);
                CK(cudaMemcpyAsync(meta.device,meta.pinned,map.size()*8,cudaMemcpyHostToDevice));
            } else CK(cudaMemcpy(meta.device,map.data(),map.size()*8,cudaMemcpyHostToDevice));
            g_gdevice.metadata_words+=map.size();
            S4DeviceBatch batch{cur,meta.device,next,map.data(),cur_words,next_words,nb};
            std::vector<unsigned long long> unused;
            poly_mul_batch_modN(L,nullptr,nullptr,ma,mb,nb,unused,cat,nullptr,0,ma+mb-1,&batch);
            ++g_gdevice.groups;g_gdevice.pairs+=nb;fs.muls+=nb;
            ++L.s4->groups;
        }
        const char *fault=std::getenv("NTT_GROOT_DEVICE_TEST_BAD");
        if(fault && std::atoi(fault) && !poisoned && next_words) {
            CK(cudaMemsetAsync(next,0xff,8));poisoned=true;
            stage2_log::print(stage2_log::debug, "gdevice_fault: first computed frontier poisoned\n");
        }
        std::swap(cur,next);offsets.swap(target);cur_words=next_words;++g_gdevice.levels;validate(base);
        if(g_s4_batched_progress)
            stage2_log::print(stage2_log::debug, "gdevice_progress: base=%llu groups=%llu t_level=%.3f t_total=%.3f\n",
                (unsigned long long)base,(unsigned long long)groups.size(),now_s()-level_start,now_s()-start);
    }
    if(root_sink) {
        // The consumer copies before any subsequent call can recycle raw A/B.
        root_sink(cur,n+1);g_gdevice.root_words+=(n+1)*W;return tree;
    }
    tree[1].resize((n+1)*W);
    const double tr=now_s();CK(cudaMemcpy(tree[1].data(),cur,tree[1].size()*8,cudaMemcpyDeviceToHost));
    L.t_d2h_coeff+=now_s()-tr;L.d2h_coeff_words+=tree[1].size();g_gdevice.root_words+=tree[1].size();
    fs.node_peak_bytes=fs.node_retained_bytes=8ull*tree[1].capacity();
    return tree;
}

std::vector<std::vector<unsigned long long>> build_groot_select(
    PolyLayer &L,
    const std::vector<std::vector<unsigned long long>> &leaf,
    std::vector<size_t> &deg,
    size_t &pad,
    FTreeStats &fs,
    int cat,
    bool keep_children,
    const std::function<void(const unsigned long long*,size_t)> &root_sink,
    bool balanced)
{
    const char *host=std::getenv("NTT_S4_HOSTPACK");
    if(g_groot_device && !keep_children && L.s4 && g_s4_pack_direct && !g_s4_final_readback &&
       !(host && std::atoi(host)))return build_groot_device(L,leaf,deg,pad,fs,cat,0,{},root_sink,balanced);
    if(g_groot_device)++g_gdevice.fallbacks;
    return build_tree_flat(L,leaf,deg,pad,fs,cat,keep_children);
}

void groot_input_digest_host(
    const std::vector<unsigned long long> &input,
    unsigned long long &words,
    unsigned long long *digest)
{
    for(const auto value:input) {
        digest[0]+=groot_mix64(value^groot_mix64(words+0x9e3779b97f4a7c15ull));
        digest[1]^=groot_mix64(value+groot_mix64(words+0xd1b54a32d192ed03ull));++words;
    }
}
