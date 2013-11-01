#pragma once
#include "../componentSys/Object.h"

namespace AI{
	class Sensor;
	class Task;
	class Executor;
	
	/*!converts an Object to an AI Agent. 
	Adds properties to the Object that is used by the entire AI framework in the game.
	It basically "initializes" the AI components in the object*/
	void makeAgent(Object *obj);


	class Sensor{
		Object *owner;

		std::string name;
	protected:
		//only a Task can instantiate this
		Sensor(Object *owner, std::string name);
		friend class Task;
		
	public:
		virtual void Sense() = 0;
		
		std::string getName() const;
		typedef std::map<const Hash *, Sensor> sensorMap;
	};


	class Task{
	private:
		Object::objectMap &objectMap;
		Object *owner;
	public:
		Task(Object::objectMap &objectMap, Object *owner);

		//return a normalized "importance" of this task
		virtual float getImportance() const = 0 ;

		//return a normalized "urgency" of this task
		virtual float getUrgency() const = 0;
		
		//return whether another task can run in parallel to this task
		//or if this task must be performed exclusively'
		bool isExclusive() const;

		//similar to what covey describes - importance and urgency - 
		//for example, at full HP, staying alive is important but not urgent.
		//Tasks are arranged according to urgency first, then according to importance.
		//so, the most urgent tasks are identified, and then the most important task among them
		//is chosen. this seems to be a good model

		//create and return the executors needed for this task
		virtual void createSensors(std::vector<Sensor *> &sensors) = 0;
		virtual void createExecutors(std::vector<Executor *> &executor) = 0;
	};

	class Executor{
	private:
		Object::objectMap &objectMap;
		Object *owner;

	protected:
		//only a Task can instantiate this
		Executor(Object::objectMap &objectMap, Object *owner);
		friend class Task;

	public:
		virtual void Execute(Task &task) = 0;

		typedef std::map<const Hash *, Executor> executorMap;
	};	

	class Brain{
		Sensor::sensorMap sensors;
		Executor::executorMap executors;
		std::vector<Task *>tasks;

		Object *owner;
		Brain();
		~Brain();


	};
}
