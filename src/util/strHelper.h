#pragma once
#include <string>
#include <sstream>

namespace util{
	class strHelper{
	private:
		static std::stringstream sStream;
	public:
		template<typename T>
		static std::string toStr(const T &value){
			sStream.clear();
			sStream<<value;
			return sStream.str();
		}
	};
};