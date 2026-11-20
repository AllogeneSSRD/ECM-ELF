#include "ecm/io/results.h"
#include "io/json.h"
#include "io/files.h"
#include "ecm/io/save.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <limits>
namespace fs=std::filesystem;namespace r=ecm_results;
void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}
std::string bytes(const fs::path&p){std::ifstream in(p,std::ios::binary);return {std::istreambuf_iterator<char>(in),{}};}
void text(const fs::path&p,const std::string&s){std::ofstream out(p,std::ios::binary|std::ios::trunc);out<<s;}
std::vector<r::json::Value> rows(const fs::path&p){std::vector<r::json::Value> result;std::istringstream in(bytes(p));std::string line;while(std::getline(in,line))if(!line.empty())result.push_back(r::json::parse(line));return result;}
fs::path block_receipt(const fs::path& output, const std::string& key) {
 const auto pending=output.parent_path()/"data"/(output.filename().string()+".ecm-publish")/
     "receipts"/(std::to_string(ecm_io::fingerprint(key))+".done.new");
 fs::create_directories(pending);
 return pending;
}
std::vector<r::Stage1BatchRecord> batch(mpz_srcptr n,const std::vector<uint64_t>&sigmas,bool factor=false) {
 std::vector<r::Stage1BatchRecord> out;
 for(auto sigma:sigmas) {
  r::Submission s;s.exponent=29;s.stage1_only=true;s.b1=s.b2=20;s.sigma=sigma;s.when="2026-10-24 12:00:00";
  if(factor && sigma!=20){s.stage=1;s.factors={"233"};}
  out.push_back({s,r::internal_json(s,n,0,"batch-contract",",\"stage1-engine\":{\"coverage\":\"B1\"}")});
 }
 return out;
}
int main(int argc,char **argv) {
 try {
  check(argc==2,"output directory required");fs::path dir=argv[1];check(!fs::exists(dir),"fresh directory required");fs::create_directories(dir);
  ecm_edwards::Integer n;mpz_set_ui(n.v,1);mpz_mul_2exp(n.v,n.v,29);mpz_sub_ui(n.v,n.v,1);
  const auto internal=dir/"stage1_results.jsonl",submission=dir/"results.json.txt";
  r::publish_stage1_batch(internal,submission,batch(n.v,{100,3,20},true),n.v,"batch-a");
  auto values=rows(internal);check(values.size()==3,"internal count");
  check(values[0].get("Edwards").get("sigma").text()=="3" && values[1].get("Edwards").get("sigma").text()=="20" &&
        values[2].get("Edwards").get("sigma").text()=="100","numeric sigma ordering");
  auto public_rows=rows(submission);check(public_rows.size()==3,"two factor and one NF");
  check(public_rows[0].get("Edwards").get("sigma").text()=="3" && public_rows[1].get("Edwards").get("sigma").text()=="100","factor order");
  check(public_rows[2].get("curves").text()=="1" && public_rows[2].get("status").text()=="NF","NF aggregation");
  check(!r::has_result_ledger(submission),"Stage1 created a legacy ledger");
  auto a=bytes(internal),b=bytes(submission);
  r::publish_stage1_batch(internal,submission,batch(n.v,{20,100,3},true),n.v,"batch-a");
  check(bytes(internal)==a && bytes(submission)==b,"retry duplicated results");
  text(submission,b.substr(0,b.size()-10));
  r::publish_stage1_batch(internal,submission,batch(n.v,{100,20,3},true),n.v,"batch-a");
  check(bytes(internal)==a && bytes(submission)==b.substr(0,b.size()-10),"committed retry rewrote archived output");
  fs::remove(submission);
  r::publish_stage1_batch(internal,submission,batch(n.v,{3,20,100},true),n.v,"batch-a");
  check(!fs::exists(submission),"committed retry recreated archived output");
  text(submission,b);
  const auto relocated=dir/"relocated";fs::create_directory(relocated);
  fs::copy_file(internal,relocated/internal.filename());fs::copy_file(submission,relocated/submission.filename());
  fs::copy(dir/"data",relocated/"data",fs::copy_options::recursive);
  r::publish_stage1_batch(relocated/internal.filename(),relocated/submission.filename(),batch(n.v,{3,20,100},true),n.v,"batch-a");
  check(bytes(relocated/internal.filename())==a && bytes(relocated/submission.filename())==b,"relocated pair retry");
  const auto failure_internal=dir/"failure.jsonl",failure_public=dir/"failure.txt";
  fs::create_directory(fs::path(failure_public.string()+".new"));bool write_failed=false;
  try{r::publish_stage1_batch(failure_internal,failure_public,batch(n.v,{31,21}),n.v,"write-failure");}catch(const std::runtime_error&){write_failed=true;}
  check(write_failed && fs::exists(failure_internal) && !fs::exists(failure_public),"second output failure was not retained");
  const auto wrong_public=dir/"wrong-destination.txt";
  bool wrong_path=false;
  try{r::publish_stage1_batch(failure_internal,wrong_public,batch(n.v,{31,21}),n.v,"write-failure");}
  catch(const std::runtime_error& error){wrong_path=std::string(error.what()).find("different submission path")!=std::string::npos;}
  check(wrong_path && !fs::exists(wrong_public),"pending Stage1 append lost on path change");
  fs::remove(fs::path(failure_public.string()+".new"));
  r::publish_stage1_batch(failure_internal,failure_public,batch(n.v,{21,31}),n.v,"write-failure");
  check(rows(failure_internal).size()==2 && rows(failure_public).size()==1,"failed write recovery duplicated batch");
  const auto applied_internal=dir/"applied.jsonl",applied_public=dir/"applied.txt";
  const std::string archived_prefix="archived foreign line\r\n";
  text(applied_public,archived_prefix);
  const auto blocked=block_receipt(applied_public,"stage1-nf|applied-stage1");
  bool after_write=false;
  try{r::publish_stage1_batch(applied_internal,applied_public,batch(n.v,{3,20},true),n.v,"applied-stage1");}
  catch(const std::runtime_error& error){after_write=std::string(error.what()).find("cannot create temporary file")!=std::string::npos;}
  const auto applied=bytes(applied_public);
  check(after_write && applied.compare(0,archived_prefix.size(),archived_prefix)==0 &&
        applied.find("\"status\":\"F\"")!=applied.npos && applied.find("\"status\":\"NF\"")!=applied.npos,
        "Stage1 fault did not land after the atomic append");
  fs::remove(blocked);
  r::publish_stage1_batch(applied_internal,applied_public,batch(n.v,{3,20},true),n.v,"applied-stage1");
  check(bytes(applied_public)==applied,"Stage1 post-append retry duplicated a factor or NF");
  const auto original_internal=bytes(failure_internal),original_public=bytes(failure_public);
  auto damaged=rows(failure_internal);auto damaged_plan=damaged[0].get("stage1-publication");
  damaged_plan.set("identity",r::json::Value::string("changed"));damaged[0].set("stage1-publication",damaged_plan);
  text(failure_internal,damaged[0].dump()+"\n"+damaged[1].dump()+"\n");bool plan_rejected=false;
  try{r::publish_stage1_batch(failure_internal,failure_public,batch(n.v,{21,31}),n.v,"write-failure");}catch(const std::runtime_error&){plan_rejected=true;}
  check(plan_rejected && bytes(failure_public)==original_public,"damaged plan accepted");text(failure_internal,original_internal);
  text(failure_internal,rows(failure_internal)[0].dump()+"\n");bool incomplete_rejected=false;
  try{r::publish_stage1_batch(failure_internal,failure_public,batch(n.v,{21,31}),n.v,"write-failure");}catch(const std::runtime_error&){incomplete_rejected=true;}
  check(incomplete_rejected && bytes(failure_public)==original_public,"incomplete internal batch accepted");text(failure_internal,original_internal);
  r::publish_stage1_batch(internal,submission,batch(n.v,{1000,40,500}),n.v,"batch-b");
  check(bytes(internal).compare(0,a.size(),a)==0 && bytes(submission).compare(0,b.size(),b)==0,"previous batch bytes changed");
  check(r::contains_stage1_curves(internal,n.v,20,true,0,{500,40,1000}),"flat history lookup failed");
  check(!r::contains_stage1_curves(internal,n.v,20,true,0,{999}),"missing curve accepted");
  fs::rename(submission,dir/"submission-archive.txt");
  const std::string prefix="manual archive marker without newline";
  text(submission,prefix);
  r::publish_stage1_batch(internal,submission,batch(n.v,{1000,40,500}),n.v,"batch-b");
  check(bytes(submission)==prefix,"Stage1 retry restored an archived batch");
  r::publish_stage1_batch(internal,submission,batch(n.v,{40,600}),n.v,"overlap");
  a=bytes(internal);b=bytes(submission);
  r::publish_stage1_batch(internal,submission,batch(n.v,{600,40}),n.v,"overlap");
  check(bytes(internal)==a && bytes(submission)==b && rows(internal).size()==7 &&
        b.compare(0,prefix.size()+1,prefix+'\n')==0,"archive append or overlap retry/count");
  bool rejected=false;try{r::publish_stage1_batch(internal,submission,batch(n.v,{700,700}),n.v,"duplicate");}catch(const std::runtime_error&){rejected=true;}
  check(rejected && bytes(internal)==a,"duplicate sigma changed output");
  const auto wide=dir/"wide.jsonl",wide_public=dir/"wide.txt";
  r::publish_stage1_batch(wide,wide_public,batch(n.v,{UINT64_MAX,2,UINT64_MAX-1}),n.v,"wide");
  auto w=rows(wide);check(w[0].get("Edwards").get("sigma").text()=="2" &&
      w[2].get("Edwards").get("sigma").text()==std::to_string(UINT64_MAX),"uint64 ordering");
  const auto save=dir/"e_m29_20.save";
  check(ecm_save::publish(save.string(),ecm_edwards::save_line(n.v,n.v,100,20,false)+
       ecm_edwards::save_line(n.v,n.v,3,20,false)+ecm_edwards::save_line(n.v,n.v,20,20,false),true),"save publication failed");
  auto saved=bytes(save);check(saved.find("SIGMA=3;")<saved.find("SIGMA=20;") && saved.find("SIGMA=20;")<saved.find("SIGMA=100;"),"save ordering");
  const auto json_save=dir/"cpu_stage2.jsonl";
  r::append_text_record(json_save,"{\"sigma\":100}");
  r::append_text_record(json_save,"{\"sigma\":3}");
  check(bytes(json_save)=="{\"sigma\":100}\n{\"sigma\":3}\n","CPU Stage2 append was reordered");
  r::Submission stage2;stage2.exponent=29;stage2.sigma=30;stage2.b1=20;stage2.b2=2000;
  stage2.publication_task="stage2-a";stage2.save_name="e_m29_20";
  const auto s2=dir/"stage2"/"results.json.txt";
  r::publish_submission(s2,stage2,n.v,false);stage2.sigma=40;r::publish_submission(s2,stage2,n.v,false);
  const auto ledger=dir/"stage2"/"data"/"results.json.txt.ecm-ledger";
  check(fs::exists(ledger/"e_m29_20"),"Stage2 save namespace missing");
  size_t count=0;for(const auto&f:fs::directory_iterator(ledger/"e_m29_20"))if(f.path().extension()==".curve")++count;
  check(count==2,"scoped receipt count");
  r::seal_submissions(s2,2,"stage2-a",true);check(rows(s2).size()==1 && rows(s2)[0].get("curves").text()=="2","scoped seal");
  const auto first=bytes(s2);stage2.save_name="another_save";stage2.sigma=30;r::publish_submission(s2,stage2,n.v,false);
  r::seal_submissions(s2,2,"stage2-a",true);check(bytes(s2)==first,"same curve counted under two saves");
  fs::rename(ledger,dir/"archived_stage2_ledger");
  fs::rename(s2,dir/"archived_stage2_submission");
  text(s2,prefix);
  r::check_stage2_outputs(dir/"missing_internal.jsonl",s2);
  stage2.publication_task="stage2-after-archive"; stage2.sigma=41;
  r::publish_submission(s2,stage2,n.v);
  r::seal_submissions(s2,2,stage2.publication_task,true);
  const auto direct=bytes(s2);
  check(direct.compare(0,prefix.size()+1,prefix+'\n')==0 &&
        r::json::parse(direct.substr(prefix.size()+1)).get("curves").text()=="1",
        "direct Stage2 did not append after archive");
  r::publish_submission(s2,stage2,n.v);
  r::seal_submissions(s2,2,stage2.publication_task,true);
  check(bytes(s2)==direct,"direct Stage2 retried NF after archive");
  const auto pending_internal=dir/"pending_stage2.jsonl",pending_public=dir/"pending_stage2.txt";
  r::publish_stage1_batch(dir/"pending_stage1.jsonl",pending_public,batch(n.v,{300}),n.v,"pending-stage1");
  r::check_stage2_outputs(pending_internal,pending_public);
  stage2.sigma=300;stage2.save_name="pending_save";r::publish_submission(pending_public,stage2,n.v,false);
  text(pending_internal,r::internal_json(stage2,n.v,0,"loss-contract",",\"stage2-engine\":{}"));
  fs::rename(dir/"data/pending_stage2.txt.ecm-ledger",dir/"lost_pending_ledger_evidence");
  r::check_stage2_outputs(pending_internal,pending_public);
  r::check_stage2_outputs(dir/"missing.jsonl",pending_public,true);
  // A stale legacy export cannot veto normal queue publication.
  const auto checkpoint=dir/"checkpoint.txt";
  text(checkpoint,archived_prefix);
  const auto legacy_export=r::submission_ledger(checkpoint);
  fs::create_directories(legacy_export);
  text(legacy_export/"format","ECM_SUBMISSION_LEDGER_V1\n");
  text(legacy_export/"publication.order","damaged legacy publication history");
  auto factor=stage2;factor.stage=2;factor.factors={"233"};
  const auto hit=r::submission_json(factor,n.v), latest=r::submission_json(stage2,n.v);
  std::string final_nf;
  const auto after_receipt=block_receipt(checkpoint,"stage2-nf|checkpoint-stage2");
  after_write=false;
  try{r::publish_stage2_checkpoint(checkpoint,{hit},latest,2,&final_nf,[]{}, "checkpoint-stage2");}
  catch(const std::runtime_error& error){after_write=std::string(error.what()).find("cannot create temporary file")!=std::string::npos;}
  const auto checkpoint_written=bytes(checkpoint);
  check(after_write && checkpoint_written==archived_prefix+hit+final_nf+'\n',
        "Stage2 fault did not land after the atomic append");
  fs::remove(after_receipt);
  r::publish_stage2_checkpoint(checkpoint,{hit},latest,2,&final_nf,[]{}, "checkpoint-stage2");
  check(bytes(checkpoint)==checkpoint_written,"Stage2 post-append retry duplicated a factor or NF");
  fs::rename(checkpoint,dir/"checkpoint-archive.txt");
  r::publish_stage2_checkpoint(checkpoint,{hit},latest,2,&final_nf,[]{}, "checkpoint-stage2");
  check(!fs::exists(checkpoint),"Stage2 retry restored archived reports");
  // Both stages may share the submission output. Stage1 adds no ledger;
  // Stage2 adopts that prefix, then Stage1 appends after the Stage2 order file.
  const auto shared=dir/"shared.txt",shared_internal=dir/"shared.jsonl";
  r::publish_stage1_batch(shared_internal,shared,batch(n.v,{3},true),n.v,"shared-a");
  const auto shared_first=bytes(shared);
  auto f=batch(n.v,{3},true)[0].submission;f.stage1_only=false;f.b2=2000;f.save_name="shared_save";
  r::publish_submission(shared,f,n.v,false);check(bytes(shared)==shared_first,"Stage2 repeated Stage1 factor");
  stage2.sigma=50;stage2.save_name="shared_save";stage2.publication_task="shared-stage2";
  r::publish_submission(shared,stage2,n.v,false);r::seal_submissions(shared,2,"shared-stage2",true);
  auto shared_prefix=bytes(shared);
  r::publish_stage1_batch(shared_internal,shared,batch(n.v,{9,8}),n.v,"shared-b");
  auto shared_after=bytes(shared);check(shared_after.compare(0,shared_prefix.size(),shared_prefix)==0,"shared prefix changed");
  r::refresh_submission_if_owned(shared);check(bytes(shared)==shared_after,"Stage2 could not adopt Stage1 suffix");
  r::publish_stage1_batch(shared_internal,shared,batch(n.v,{8,9}),n.v,"shared-b");
  check(bytes(shared)==shared_after,"shared retry changed output");
  // An old immediate factor can exist before its internal batch is rendered.
  const auto legacy=dir/"legacy.txt",legacy_internal=dir/"legacy.jsonl";
  f=batch(n.v,{3},true)[0].submission;
  r::publish_submission(legacy,f,n.v,false);auto legacy_before=bytes(legacy);
  r::publish_stage1_batch(legacy_internal,legacy,batch(n.v,{3,20},true),n.v,"legacy-retry");
  check(rows(legacy).size()==2 && bytes(legacy).compare(0,legacy_before.size(),legacy_before)==0,"legacy factor duplicated");
  for(unsigned param:{0u,2u,3u}) {
      auto mont=batch(n.v,{100,3,20});for(auto &record:mont){record.submission.edwards=false;record.submission.param=int(param);
          record.internal=r::internal_json(record.submission,n.v,0,"mont-contract",",\"stage1-engine\":{}");}
      const auto m=dir/("mont"+std::to_string(param)+".jsonl"),p=dir/("mont"+std::to_string(param)+".txt");
      r::publish_stage1_batch(m,p,mont,n.v,"mont");
      check(rows(m)[0].get("sigma").text()=="3" && r::contains_stage1_curves(m,n.v,20,false,param,{3,20,100}),"Montgomery history/order");
  }
  std::cout<<"result_batch_contract: passed=1\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
