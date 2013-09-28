#pragma once
#include "Object.h"
#include "processor/objectProcessor.h"

#include <vector>
#include "../IO/logObject.h"


/*!Manages Object class's lifecycle
the ObjectMgr is in charge of controlling Object and running
objectProcessor on the objects. It's the heart of the component based 
system present in the engine

\sa Object
\sa objectProcessor
\sa objectMgrProc
*/
class objectMgr{

private:
	
	Object::objectMap objMap;
	Object::objectMap activeObjects;

	std::vector<objectProcessor *> objProcessors;
	typedef std::vector<objectProcessor *>::iterator objProcessorIt;
public:
	objectMgr(){};
	~objectMgr(){};

	/*!add an Object to the objectManager for processing
	once an Object is added, it is updated and drawn every frame 
	until the Object dies. 
	
	@param [in] obj the Object to be added
	*/
	void addObject(Object *obj);

	/*!deactivate an Object thereby pausing interaction */
	void deactivateObject(Object &obj);
	void deactivateObject(const char* name);

	/*!activate an Object thereby allowing it to interact */
	void activateObject(Object &obj);
	void activateObject(const char* name);

	/*!get an object by it's name*/
	Object *getObjByName(const char* name);

	/*!add an objectProcessor to make it process Objects
	@param [in] processor the objectProcessor to be added
	*/
	void addObjectProcessor(objectProcessor *processor);

	/*!remove an objectProcessor that had been added
	@param [in] processor the objectProcessor to be removed 
	*/
	void removeObjectProcessor(objectProcessor *processor);

	/*!pre-processing takes place
	generally, variables are setup if need be in this step, 
	and everything is made ready for Process
	*/
	void preProcess();

	/*!cleaning up takes place
	once Process is called, postProcess takes care of cleaning
	behind all of the leftover variables and data.
	Dead Objects are also destroyed in this step
	*/
	void postProcess();

	/*!Processes all objects*/
	void Process(float dt);
};
