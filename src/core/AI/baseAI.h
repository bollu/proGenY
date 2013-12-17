#pragma once
#include "../componentSys/Object.h"
#include <functional>

namespace AI{
	typedef std::function<void(Object*)> ThinkFunc;
	
}
