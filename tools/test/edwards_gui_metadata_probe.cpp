#include "log_parse.h"
#include "results.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Replay actual worker pipe output through the same pure status/hit projection
// used by WorkerProc, then through the real ResultsStore JSONL writer.
int main(int argc,char **argv) {
    try {
        if(argc!=5)throw std::runtime_error("usage: edwards_gui_metadata_probe STAGE LOG OUTPUT_JSON EXPECTED_MODEL");
        const std::string stage=argv[1],model=argv[4];
        if(stage!="stage1" && stage!="stage2")throw std::runtime_error("invalid stage");
        if(std::filesystem::exists(argv[3]))throw std::runtime_error("use a fresh output JSON file");
        std::ifstream input(argv[2]);if(!input)throw std::runtime_error("cannot read worker log");
        ecmgui::ResultsStore store;std::string error;
        if(!store.init(argv[3],std::string(argv[3])+".txt",error))throw std::runtime_error(error);
        ecmgui::Stage2Status state;unsigned count=0;
        auto add=[&](const ecmgui::HitInfo &hit) {
            if(hit.curve_model!="ed" && hit.curve_model!="mont")throw std::runtime_error("missing curve identity");
            if(model!="mixed" && hit.curve_model!=model)throw std::runtime_error("wrong curve identity");
            if((hit.curve_model=="ed" && hit.param!=-1) || (hit.curve_model=="mont" && hit.param<0))
                throw std::runtime_error("wrong Montgomery PARAM presence");
            if(!hit.has_sigma || hit.save.empty())throw std::runtime_error("missing sigma or save");
            ecmgui::HitRecord record;record.factor=hit.factor;record.curve=hit.curve;
            record.sigma=hit.sigma;record.has_sigma=hit.has_sigma;record.save=hit.save;
            record.param=hit.param;record.curve_model=hit.curve_model;record.execution_target=hit.execution_target;
            if(!store.add(record,error))throw std::runtime_error(error);
            ++count;
        };
        std::string line;
        while(std::getline(input,line)) {
            const auto parsed=ecmgui::parse_line(line);
            if(stage=="stage1") {
                if(parsed.is_hit && !parsed.hit.curve_model.empty())add(parsed.hit);
            } else if(parsed.stage2.kind!=ecmgui::Stage2Event::Kind::None) {
                state.apply(parsed.stage2);
                for(const auto &factor:parsed.stage2.factors)add(state.factor_hit(factor));
            }
        }
        if(!count)throw std::runtime_error("no worker factors were replayed");
        if(!store.flush(error) || !store.rebuild_from_jsonl(error))throw std::runtime_error(error);
        std::cout<<"PASS worker pipe -> GUI status -> results JSONL: hits="<<count<<'\n';return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
