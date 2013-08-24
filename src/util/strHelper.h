#pragma once
#include <string>
#include <sstream>

namespace util{

	/*!contains useful string manipulation functions*/
	class strHelper{
	private:
		//static std::stringstream sStream;
	public:

		/*!helper function to convert any type to a string. 
		Useful for debugging

		@param [in] value the value to be converted to a string

		\return the stringified form of the value
		*/ 
		template<typename T>
		static std::string toStr(const T &value){
			//sStream.clear();
			std::stringstream sStream;
			sStream<<value;
			return sStream.str();
		}
	};
};