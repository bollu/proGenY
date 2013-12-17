#pragma once
#include <algorithm>

template <typename T=float>
class Cooldown {
public:
	Cooldown(T totalTime) : totalTime_(totalTime), currentCooldown_(0) {}
	Cooldown() : totalTime_(0), currentCooldown_(0) {}

	void setTotalTime(T totalTime) {
		totalTime_ = totalTime;
		currentCooldown_ = 0;
	}

	void startCooldown() { currentCooldown_ = totalTime_; }

	Cooldown& Tick(T dt) { 
		if(currentCooldown_ > 0) currentCooldown_ = std::max<T>(currentCooldown_ - dt, 0.0);  else currentCooldown_ = 0;;

		return (*this);
	}
	
	

	bool onCooldown() { return currentCooldown_ > 0; }
	bool offCooldown() {return currentCooldown_ == 0; }
private:
	T totalTime_;
	T currentCooldown_;
};