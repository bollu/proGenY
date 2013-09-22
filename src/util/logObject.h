#pragma once
#include <iostream>
#include <assert.h>
#include "strHelper.h"
#include <typeinfo>
#include "../core/Hash.h"

/*!
	@file logObject.h
	Logging objects are present on this file

*/


namespace util{
	/*! \enum logLevel
		an Enum of various Logging levels available 
	*/
	enum logLevel{
		/*! Information. used for debugging / printing to the console*/
		logLevelInfo = 0, 
		/*! Warnings. The program can continue running, but is not ideal */
		logLevelWarning, 
		/*! Errors. The program will print the error to the console and halt execution_ */
		logLevelError, 
		/*! The log level to be set that will ensure that no log message_ will be emitted */
		logLevelNoEmit, 
	};

	/*! a base class used to represent Logging objects 
		\sa msgLog scopedLog
	*/
	class baseLog{
	protected:
		baseLog();
			//only logs that are >= threshold level are emitted
		static logLevel thresholdLevel; 

	public:

		/*! only logs that have a logLevel greater than or equal to thresholdLevel are emitted.
		
		Use this function to set a threshold level for logObjects. only logObjects whose
		logLevels are greater than or equal to the threshold level. This can be used to turn off
		info and warning logs during Release builds.

		@param [in] logThreshold the logLevel that acts as the threshold for all logObjects.
		*/     
		static void setThreshold(logLevel logThreshold){
			baseLog::thresholdLevel = logThreshold;
		}

		virtual ~baseLog();
	};



	/*sentinel class */

	class _Flush{ public: _Flush(){}; };
	static const _Flush flush;

	/*! used to emit a quick logging message

	this logObject is typically used to print information to the console.
	if a logLevel of Error is used, then the program will halt after printing
	the error to the console.

	Otherwise, the message will be printed to the console and the program will
	continue to execute
	*/

	template <logLevel level>
	class msgLog : public baseLog{
	
	public:

		msgLog(){};

		template <typename T>
		msgLog & operator << (const T &toWrite){
			if(level >= baseLog::thresholdLevel){
				std::cout<<toWrite;
			}

			return *this;
		}


		msgLog & operator << (const Hash* toWrite){
			if(level >= baseLog::thresholdLevel){
				std::cout<<Hash::Hash2Str(toWrite);
			}

			return *this;
		}

		msgLog & operator << (const _Flush& flush){
			if(level >= baseLog::thresholdLevel){
				std::cout.flush();
			}
			return *this;
		}
	
		
	};

	template<>
	class msgLog<logLevelError> : public baseLog{
	private:
		void Throw(){
			std::cout<<"\n\n-----------------------------"<<std::endl;
			assert(false);
		}

	public:
		msgLog(){
		};

		template <typename T>
		msgLog & operator << (const T &toWrite){
			if(logLevelError >= baseLog::thresholdLevel){
				std::cout<<toWrite;
			}

			return *this;
		}


		msgLog & operator << (const Hash* toWrite){
			if(logLevelError >= baseLog::thresholdLevel){
				std::cout<<Hash::Hash2Str(toWrite);
			}

			return *this;
		}

		msgLog & operator << (const _Flush& flush){
			if(logLevelError >= baseLog::thresholdLevel){
				std::cout.flush();
				this->Throw();
			}
			
			return *this;
		}
	};

	#pragma once
	#ifndef LOG_GLOBAL_OBJECTS
	#define LOG_GLOBAL_OBJECTS
	static util::msgLog<logLevelInfo> infoLog;
	static util::msgLog<logLevelError> errorLog;
	static 	util::msgLog<logLevelWarning> warningLog;
	#endif



	/*! used to emit messages when a scope is entered and exited 

	the scopedLog can be used to quickly chart out scope by creating a 
	scopedLog at the beginning of a block. when the block ends,
	the scopedLog is destroyed and the required message is emitted.

	it _must_ be created on the stack for it to send the destruction message.
	if created on the heap, it will send the destruction message when it is destroyed, 
	which is rarely (if at all) useful

	\sa msgLog
	*/ 
	class scopedLog : public baseLog{
	private:
		std::string onDestroyMsg;
		bool enabled;
	public:
		/*! constructor used to create a scopedLog

		@param [in] onCreateMsg message to send when created
		@param [in] onDestroyMsg message to send when destroyed
		@param [in] level the logging level of this message

		*/ 
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
