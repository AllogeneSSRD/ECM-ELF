#pragma once
#include <cmath>
#include <cstdint>
#include <random>

namespace ecmgui_v2::pet {
// Blink timings and forelock key times adapted from ELYTerminal look/curves.rs.
// Copyright (c) 2026 ELYTerminal contributors, MIT; see third_party/gui_v2_pet/LICENSE.
enum class Mood { Calm, Working, Waiting, Error, Done, Doze, Wink };
enum class Gesture { None, Blink, Forelock };
struct Pose { Mood mood=Mood::Calm; int frame=0; float rise=0,forelock=0; Gesture gesture=Gesture::None; };
// Monotonic milliseconds in, presentation out. No window, clock or renderer access.
class Behavior {
public:
    static constexpr uint64_t sleep_after_ms=300000;
    explicit Behavior(uint32_t seed=1):random_(seed){}
    void interact(uint64_t now,bool affection=true){
        start(now);last_interaction_=now;affection_until_=affection?now+1500:0;
        gesture_=Gesture::None;schedule(now);
    }
    Pose sample(uint64_t now,Mood source,bool animate=true,bool dragging=false){
        start(now);
        if(source!=source_){source_=source;source_since_=now;}
        Pose pose;pose.mood=source;
        if(source==Mood::Done&&now-source_since_>=4000)pose.mood=Mood::Calm;
        if(now-last_interaction_>=sleep_after_ms&&pose.mood==Mood::Calm&&!dragging)pose.mood=Mood::Doze;
        if(now<affection_until_)pose.mood=Mood::Wink;
        if(dragging){last_interaction_=now;pose.mood=source;}
        if(!animate||dragging||pose.mood==Mood::Doze||pose.mood==Mood::Error||pose.mood==Mood::Wink){
            gesture_=Gesture::None;paused_=true;
            if(animate&&!dragging&&pose.mood==Mood::Wink)pose.frame=int((now-last_interaction_)/220)%8;
            return pose;
        }
        if(paused_){schedule(now);paused_=false;}
        // Absolute time drives the motion, so extra render frames do not speed it up.
        pose.rise=float(1.5*(1-std::cos(double(now-origin_)*6.283185307179586/7000.0)));
        if(gesture_!=Gesture::None&&now-gesture_since_>=duration())gesture_=Gesture::None;
        if(gesture_==Gesture::None){
            if(now>=next_blink_){gesture_=Gesture::Blink;gesture_since_=now;next_blink_=now+blink_delay();}
            else if(now>=next_forelock_){gesture_=Gesture::Forelock;gesture_since_=now;next_forelock_=now+delay(18000,22000);}
        }
        pose.gesture=gesture_;
        if(gesture_==Gesture::Blink)pose.frame=7;
        if(gesture_==Gesture::Forelock)pose.forelock=wiggle(now-gesture_since_);
        return pose;
    }
private:
    std::mt19937 random_;bool initialized_=false,paused_=false;
    uint64_t origin_=0,last_interaction_=0,affection_until_=0,source_since_=0;
    uint64_t next_blink_=0,next_forelock_=0,gesture_since_=0;
    Mood source_=Mood::Calm;Gesture gesture_=Gesture::None;
    unsigned blink_index_=0;
    uint64_t delay(int low,int high){return std::uniform_int_distribution<int>(low,high)(random_);}
    uint64_t blink_delay(){constexpr int gaps[]={3400,4600,320,5200,2800,4100};const int gap=gaps[blink_index_++%6];return delay(gap-gap/10,gap+gap/10);}
    static float wiggle(uint64_t t){
        constexpr int times[]={0,120,280,460,700};constexpr float values[]={0,-1,.6875f,-.375f,0};
        for(int i=0;i<4;++i)if(t<unsigned(times[i+1])){float k=float(t-times[i])/(times[i+1]-times[i]);k=k*k*(3-2*k);return values[i]+(values[i+1]-values[i])*k;}return 0;
    }
    uint64_t duration()const{return gesture_==Gesture::Blink?220:700;}
    void schedule(uint64_t now){blink_index_=0;next_blink_=now+blink_delay();next_forelock_=now+delay(11000,13800);}
    void start(uint64_t now){if(!initialized_){initialized_=true;origin_=last_interaction_=source_since_=now;schedule(now);}}
};
}
