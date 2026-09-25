#include <franka_governor/governor.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <string>
using namespace franka_governor;
static size_t allocations=0;
void* operator new(std::size_t n) { ++allocations; if(void* p=std::malloc(n))return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
#define CHECK(x) do { if(!(x)) { std::cerr<<"check failed line "<<__LINE__<<"\n";return 1;} } while(false)
Config config() {
 Config c;
 c.lower={-2.9007400166666666,-1.8360900166666667,-2.9007400166666666,-3.077020016666667,-2.87630335,.43982265,-3.05083335};
 c.upper={2.9007400166666666,1.8360900166666667,2.9007400166666666,-.11693708333333333,2.87630335,4.62163335,3.05083335};
 c.envelope_velocity={2.62,2.62,2.62,2.62,5.26,4.18,5.26};
 c.envelope_offset={.6520000381679385,.2499685851463105,.20053055661577582,.3542181933842238,.5738321197718637,.4884649342105264,.4591952376425851};
 c.envelope_deceleration={6.,2.585,3.5,4.,17.,5.5,17.};
 c.margin.fill(.1);c.max_velocity.fill(.3);c.max_acceleration.fill(2.);c.max_jerk.fill(30.);
 c.default_position={0,-.569,0,-2.81,0,3.037,.741};c.tracking_error.fill(.2);c.desired_error.fill(.001);c.desired_velocity_error.fill(.01);c.desired_acceleration_error.fill(.1);
 c.max_segment_distance=.03;c.max_projection=4.;c.horizon_ticks=250;c.max_blocked_ticks=1000;
 c.max_projected_ticks=5000;c.action_timeout_ns=100000000;c.observation_timeout_ns=10000000;c.tick_tolerance_ns=0;
 return c;
}
Feedback feedback(const Output& o,std::int64_t t) {
 Feedback f;f.session=1;f.observed_ns=t;f.q=f.desired_q=o.q;f.dq=f.desired_dq=o.dq;f.desired_ddq=o.ddq;return f;
}
int main(int argc,char**) {
 Config c=config();Governor g(c);Feedback f;f.q=f.desired_q=c.default_position;f.session=1;
 CHECK(g.reset(1,0,f));const bool trace=argc>1;
 std::uint64_t seq=0;double maximum_jerk=0;
 for(int tick=1;tick<=5000;++tick) {
  auto before=g.output();const auto t=tick*tick_ns;
  const auto count=allocations;
  if(tick<=4000 && tick%34==1) {
   Message m;m.session=1;m.sequence=++seq;m.observation_ns=m.completed_ns=t;
   for(int j=0;j<7;++j)m.action[j]=(tick<2000?.2f:-.2f);
   CHECK(g.submit(m,t));
  }
  auto o=g.step(t,feedback(before,t));
  CHECK(allocations==count);
  CHECK(o.command_valid);CHECK(o.status!=Status::Fault);
  for(int j=0;j<7;++j) {
   CHECK(o.q[j]>=c.lower[j]+c.margin[j] && o.q[j]<=c.upper[j]-c.margin[j]);
   CHECK(std::abs(o.dq[j])<=c.max_velocity[j]);CHECK(std::abs(o.ddq[j])<=c.max_acceleration[j]);
   const double jerk=std::abs(o.ddq[j]-before.ddq[j])/dt;
   maximum_jerk=std::max(maximum_jerk,jerk);CHECK(jerk<=c.max_jerk[j]+1e-8);
  }
  if(trace && tick%100==0) {
   std::cout<<std::setprecision(17)<<tick;
   for(double x:o.q)std::cout<<','<<x;
   for(double x:o.dq)std::cout<<','<<x;
   for(double x:o.ddq)std::cout<<','<<x;
   std::cout<<','<<int(o.status)<<','<<int(o.reason)<<'\n';
  }
 }
 CHECK(g.output().status==Status::Terminal);CHECK(g.output().reason==Reason::StaleAction);
 for(double v:g.output().dq)CHECK(std::abs(v)<1e-12);
 if(!trace)std::cout<<"5000 ticks: trajectory bounds, stale stop, and zero core allocations passed; max jerk "<<maximum_jerk<<'\n';
 return 0;
}
