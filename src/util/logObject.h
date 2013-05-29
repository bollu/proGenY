#pragma once
#include <iostream>
#include <assert.h>

namespace util{
	enum logLevel{
		logLevelInfo = 0,
		logLevelWarning,
		logLevelError,
		logLevelNoEmit,
	};

	class baseLog{
	protected:
		baseLog();
			//only logs that are >= threshold level are emitted
		static logLevel thresholdLevel; 

	public:
		virtual ~baseLog();
	};

	/* emits a quick logging message */
	class msgLog : public baseLog{
	private:

	public:
		msgLog(std::string msg, logLevel level = logLevelInfo){
			if(level >= baseLog::thresholdLevel){
				std::cout<<"\n"<<msg<<std::endl;

				if(level == logLevelError){
					std::cout<<"\n\n Quitting from logObject due to error"<<std::endl;
					assert(false);
				}
			}

			
		};
	};


	/* emits a message when created, and when destroyed */
	class scopedLog : public baseLog{
	private:
		std::string onDestroyMsg;
		bool enabled;
	public:
		scopedLog(std::string onCreateMsg, std::string onDestroyMsg, logLevel level = logLevelInfo){

			enabled = (level >= baseLog::thresholdLevel);
			
			this->onDestroyMsg = onDestroyMsg;

			if(enabled){
				std::cout<<"\n"<<onCreateMsg<<std::endl;
			}



		}

		~scopedLog(){
			if(enabled){
				std::cout<<"\n"<<onDestroyMsg<<std::endl;
			}
		}
	}; //end scopedLog


} //end namespace
