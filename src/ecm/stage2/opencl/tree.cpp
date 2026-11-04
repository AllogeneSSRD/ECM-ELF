#include "tree.h"
#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>

namespace ecm_opencl::stage2 {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
using Group=std::vector<std::size_t>;
// One dispatcher controls both the batched path and the serial A/B reference.
// Dependence order is unchanged; only independent nodes within a level group.
template<class Visit>void chunks(Arithmetic &a,const Group &group,Visit visit) {
    const auto width=a.group_batches()?group.size():1;
    for(std::size_t at=0;at<group.size();at+=width)visit(at,std::min(width,group.size()-at));
}
}
ProductTree product_tree(Arithmetic &a,const Polynomial &roots,bool keep_children,GTreeTiming* timing) {
    auto mark=timing?Clock::now():Clock::time_point{};
    if(!roots.size || roots.batch!=1)throw std::invalid_argument("product tree root shape");
    ProductTree t;t.leaves=roots.size;t.padded=1;
    if(t.leaves>(std::size_t(1)<<27))throw std::invalid_argument("product tree too large");
    while(t.padded<t.leaves)t.padded*=2;t.nodes.resize(2*t.padded);
    auto one=a.constant(1);
    auto level=t.padded;
    if(!keep_children && roots.size>=2 && a.quadratic_g_leaves()){
        level/=2;
        auto pairs=a.quadratic_factors(roots);
        for(std::size_t i=0;i<level;++i)t.nodes[level+i]=i<pairs.batch?pairs.element(i):one;
        if(roots.size%2)t.nodes[level+pairs.batch]=a.linear_factors(a.window(roots,roots.size-1,1));
    }else{
        auto factors=a.linear_factors(roots);
        for(std::size_t i=0;i<level;++i)t.nodes[level+i]=i<t.leaves?factors.element(i):one;
    }
    unsigned depth=level==t.padded?0:1;
    if(timing){timing->level[depth]+=elapsed(mark);mark=Clock::now();}
    for(std::size_t base=level/2;base;base/=2) {
        std::map<std::pair<std::size_t,std::size_t>,Group> groups;
        for(std::size_t i=base;i<2*base;++i){const auto left=t.nodes[2*i].size,right=t.nodes[2*i+1].size;
            if(left==1)t.nodes[i]=t.nodes[2*i+1];else if(right==1)t.nodes[i]=t.nodes[2*i];else groups[{left,right}].push_back(i);}
        for(const auto &g:groups)chunks(a,g.second,[&](std::size_t at,std::size_t count){
            std::vector<Polynomial> left,right;left.reserve(count);right.reserve(count);
            for(std::size_t j=0;j<count;++j){const auto node=g.second[at+j];left.push_back(t.nodes[2*node]);right.push_back(t.nodes[2*node+1]);}
            auto packed_left=a.gather(left),packed_right=a.gather(right);
            auto product=keep_children?a.multiply(packed_left,packed_right):a.multiply_g_nodes(packed_left,packed_right);
            for(std::size_t j=0;j<count;++j)t.nodes[g.second[at+j]]=product.element(j);
        });
        if(!keep_children)for(std::size_t i=2*base;i<4*base;++i)t.nodes[i]={};
        ++depth;if(timing){timing->level[depth]+=elapsed(mark);mark=Clock::now();}
    }
    return t;
}
Polynomial evaluate_scaled(Arithmetic &a,const ProductTree &t,const Polynomial &h,const Polynomial &inverse) {
    const auto p=t.leaves;
    if(!p || t.nodes.size()!=2*t.padded || t.nodes[1].size!=p+1 || h.size>p || h.batch!=1 || inverse.size<p || inverse.batch!=1)
        throw std::invalid_argument("scaled descent shape");
    // root = low_P(reverse_{P-1}(H) * inverse(reverse(F))).
    // The state at a degree-d node is the first d coefficients of H/F,
    // viewed at infinity; each child uses its sibling's reversed polynomial.
    std::vector<Polynomial> current{a.multiply(a.reshape(h,p,p-1,true),a.window(inverse,0,p),0,p)};
    for(std::size_t base=1;base<t.padded;base*=2) {
        std::vector<Polynomial> next(current.size()*2);
        std::map<std::pair<std::size_t,std::size_t>,Group> groups;
        for(std::size_t j=0;j<next.size();++j){const auto child=2*base+j,sibling=child^1;
            if(!t.nodes[child].size || !t.nodes[sibling].size)throw std::invalid_argument("scaled descent requires retained children");
            const auto d=t.nodes[child].size-1,s=t.nodes[sibling].size-1;
            if(d+s!=t.nodes[base+j/2].size-1)throw std::invalid_argument("tree degree conservation");
            if(!d)continue;if(!s)next[j]=current[j/2];else groups[{d,s}].push_back(j);}
        for(const auto &g:groups)chunks(a,g.second,[&](std::size_t at,std::size_t count){
            const auto degree=g.first.first,sibling_degree=g.first.second;
            std::vector<Polynomial> parents,siblings;parents.reserve(count);siblings.reserve(count);
            for(std::size_t j=0;j<count;++j){const auto slot=g.second[at+j];parents.push_back(current[slot/2]);siblings.push_back(t.nodes[(2*base+slot)^1]);}
            auto packed=a.gather(siblings);auto reversed=a.reshape(packed,packed.size,sibling_degree,true);
            // child = [X^s .. X^(s+d-1)] (parent * reverse(sibling)).
            auto product=a.multiply(a.gather(parents),reversed,sibling_degree,degree);
            for(std::size_t j=0;j<count;++j)next[g.second[at+j]]=product.element(j);
        });
        current.swap(next);
    }
    current.resize(p);return a.gather(current);
}
Resultant::Resultant(Arithmetic &a,const Polynomial &roots):arithmetic_(a),tree_(product_tree(a,roots)) {
    const auto &f=tree_.nodes[1];auto reversed=a.reshape(f,f.size,f.size-1,true);
    inverse_=a.inverse_series(reversed,roots.size+1);fold_=a.constant(1);
}
void Resultant::append(const Polynomial &roots,GTreeTiming* timing) {
    if(!roots.size || roots.size>tree_.leaves || roots.batch!=1)throw std::invalid_argument("giant block exceeds baby degree");
    auto mark=timing?Clock::now():Clock::time_point{};
    auto g=product_tree(arithmetic_,roots,false,timing).nodes[1];
    if(timing){timing->build+=elapsed(mark);mark=Clock::now();}
    auto product=arithmetic_.multiply(fold_,g);
    if(timing){timing->fold_product+=elapsed(mark);mark=Clock::now();}
    fold_=arithmetic_.remainder(product,tree_.nodes[1],inverse_,timing?&timing->division:nullptr);++blocks_;
    if(timing)timing->remainder+=elapsed(mark);
}
Polynomial Resultant::values(){return evaluate_scaled(arithmetic_,tree_,fold_,inverse_);}
} // namespace ecm_opencl::stage2
