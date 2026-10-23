#include "../../../src/ecm/application/gui_v2/pet/behavior.h"
#include <cassert>
#include <iostream>
using namespace ecmgui_v2::pet;
int main(){
    Behavior a(123),b(123);int blinks=0,sways=0,quiet=0;Gesture previous=Gesture::None;
    for(uint64_t now=1000;now<291000;now+=20){
        const auto p=a.sample(now,Mood::Calm),q=b.sample(now,Mood::Calm);
        assert(p.mood==q.mood&&p.frame==q.frame&&p.forelock==q.forelock&&p.rise==q.rise);
        assert(p.rise>=0&&p.rise<=3.001f&&std::abs(p.forelock)<=1);
        if(p.gesture!=previous){blinks+=p.gesture==Gesture::Blink;sways+=p.gesture==Gesture::Forelock;}
        quiet+=p.gesture==Gesture::None;previous=p.gesture;
    }
    assert(blinks>=35&&blinks<=100&&sways>=8&&sways<=25&&quiet>12000);
    assert(a.sample(300999,Mood::Calm).mood!=Mood::Doze);
    assert(a.sample(301000,Mood::Calm).mood==Mood::Doze);
    assert(a.sample(301500,Mood::Working).mood==Mood::Working);
    assert(a.sample(302000,Mood::Error).mood==Mood::Error);
    a.interact(303000);assert(a.sample(303000,Mood::Working).mood==Mood::Wink);
    assert(a.sample(304500,Mood::Working).mood==Mood::Working);
    assert(a.sample(603000,Mood::Calm).mood==Mood::Doze);
    a.interact(604000,false);assert(a.sample(604000,Mood::Calm).mood==Mood::Calm);
    assert(a.sample(904000,Mood::Working,true,true).mood==Mood::Working);
    assert(a.sample(904100,Mood::Calm).mood==Mood::Calm);
    Behavior done;done.sample(0,Mood::Done);assert(done.sample(4000,Mood::Done).mood==Mood::Calm);
    Behavior motion;motion.sample(1000,Mood::Calm);
    assert(std::abs(motion.sample(4500,Mood::Calm).rise-3)<.001f);
    assert(motion.sample(8000,Mood::Calm).rise<.001f);
    Behavior stationary;
    for(uint64_t t=0;t<300000;t+=100){const auto p=stationary.sample(t,Mood::Calm,false);assert(p.frame==0&&p.rise==0&&p.forelock==0);}
    std::cout<<"PASS: seeded gestures, quiet dwell, time-based float, sleep boundaries, wake, dragging, errors and static mode; blink="<<blinks<<" forelock="<<sways<<"\n";
}
