// Test DLL only. Never used by the normal monitor or packaged with the GUI.
#include <windows.h>
#include <atomic>
#include <cstring>
#define API extern "C" __declspec(dllexport)
static std::atomic<int> calls{0},resets{0},last_device{-1},minimum{0},maximum{0},failure{0},mode{0},swapped{0},mismatch{0},delay{0};
static const char* ids[]{"GPU-11111111-1111-1111-1111-111111111111","GPU-22222222-2222-2222-2222-222222222222"};
static int number(void* device){return int(reinterpret_cast<size_t>(device))-1;}
API void fakeConfigure(int error,int range_mode,int swap,int bad_uuid,int delay_ms){failure=error;mode=range_mode;swapped=swap;mismatch=bad_uuid;delay=delay_ms;}
API int fakeRead(int n){return n==0?calls.load():n==1?resets.load():n==2?last_device.load():n==3?minimum.load():maximum.load();}
API int nvmlInit_v2(){return 0;}
API int nvmlShutdown(){return 0;}
API int nvmlDeviceGetHandleByIndex_v2(unsigned index,void** device){if(index>1)return 6;*device=reinterpret_cast<void*>(size_t((swapped?1-int(index):int(index))+1));return 0;}
API int nvmlDeviceGetHandleByUUID(const char* uuid,void** device){for(int i=0;i<2;++i)if(std::strcmp(uuid,ids[i])==0){*device=reinterpret_cast<void*>(size_t(i+1));return 0;}return 6;}
API int nvmlDeviceGetUUID(void* device,char* value,unsigned length){const int n=number(device);if(n<0||n>1)return 6;if(length<41)return 7;std::strcpy(value,ids[mismatch?1-n:n]);return 0;}
API int nvmlDeviceGetSupportedMemoryClocks(void*,unsigned* count,unsigned* clocks){
    if(delay)Sleep(unsigned(delay.load()));
    if(mode==5){mode=0;return 999;}
    if(mode==1||mode==2)return 3;
    if(mode==3){*count=50000;return 7;}
    if(!clocks||*count<2){*count=2;return 7;}clocks[0]=10000;clocks[1]=12000;*count=2;return 0;
}
API int nvmlDeviceGetSupportedGraphicsClocks(void*,unsigned memory,unsigned* count,unsigned* clocks){
    if(!clocks||*count<3){*count=3;return 7;}
    clocks[0]=mode==4?2400:210;clocks[1]=mode==4?2450:2600;clocks[2]=mode==4?2500:memory==12000?3105:3000;*count=3;return 0;
}
API int nvmlDeviceGetMinMaxClockOfPState(void*,int type,int state,unsigned* low,unsigned* high){if(mode!=2||type||state>1)return 3;*low=state?300:210;*high=state?2200:3000;return 0;}
#ifndef NO_LOCK_EXPORTS
API int nvmlDeviceSetGpuLockedClocks(void* device,unsigned low,unsigned high){++calls;last_device=number(device);minimum=int(low);maximum=int(high);return failure.load();}
API int nvmlDeviceResetGpuLockedClocks(void* device){++resets;last_device=number(device);return failure.load();}
#endif
API const char* nvmlErrorString(int code){return code==4?"Insufficient Permissions":code==3?"Not Supported":"Fake driver error";}
