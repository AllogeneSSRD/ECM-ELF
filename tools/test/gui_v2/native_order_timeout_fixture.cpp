#include <windows.h>
#include <filesystem>
#include <fstream>
// Process-control fixture, not a mathematical oracle. Its p=101 component is
// complete while the fake residual operation deliberately never returns.
int wmain(int argc,wchar_t**argv){
    if(argc!=3)return 2;
    std::ofstream out(std::filesystem::path(argv[2]),std::ios::binary|std::ios::trunc);
    out<<R"({"protocol":1,"backend":"native","status":"partial","reason":"fixture residual operation","input_factors":[{"prime":"101","exponent":1,"primality":"proven"}],"input_cofactor":"103","components":[{"prime":"101","exponent":1,"status":"complete","group_computed":true,"point_minimality":true,"group_order":"108","point_order":"54"}]})";
    out.close();Sleep(30000);return 0;
}
