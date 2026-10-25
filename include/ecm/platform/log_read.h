#pragma once
#include <string>
#include <functional>
// Snapshot external log lines; business interpretation belongs to the caller.
// Missing/unreadable logs are reported as false, not an empty successful read.
bool ecm_visit_log_lines(const std::string &path,const std::function<void(const std::string &)> &visitor);
