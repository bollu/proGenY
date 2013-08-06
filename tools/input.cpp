#pragma once
#include "eventProcess.h"
#include "../Property.h"


void eventProcess::preUpdate(){

	while(window->pollEvent(event)){
		this->_handleEvent();
	}
};


void eventProcess::_handleEvent(){
	switch(this->event.type){
		case sf::Event::Closed:
		this->_handleWindowCloseEvent();
		break;

		case sf::Event::MouseButtonPressed:
		this->_handleMouseButtonPressed();
		break;

		case sf::Event::MouseButtonReleased:
		this->_handleMouseButtonReleased();
		break;

		case sf::Event::MouseMoved:
		this->_handleMouseMove();
		break;

		case sf::Event::KeyPressed:
		this->_handleKeyboardPressed();
		break;

		case sf::Event::KeyReleased:
		this->_handleKeyboardReleased();
		break;

		case sf::Event::MouseWheelMoved:
		this->_handleMouseWheelMove();
		break;
	}
};

void eventProcess::_handleWindowCloseEvent(){
	static const Hash *windowClosed = Hash::getHash("windowClosed"); 

	eventManager.sendEvent(windowClosed);
//	window->close(); //<- HACK for now
};


void eventProcess::_handleMouseButtonPressed(){
	static const Hash *mouseLeftPressed = Hash::getHash("mouseLeftPressedScreen"); 
	static const Hash *mouseRightPressed = Hash::getHash("mouseRightPressedScreen"); 
	

	
	vector2 mousePos = vector2(event.mouseButton.x, event.mouseButton.y);
	
	if(event.mouseButton.button == sf::Mouse::Button::Left){
		eventManager.sendEvent(mouseLeftPressed, mousePos);
	}
	else if(event.mouseButton.button == sf::Mouse::Button::Right){
		eventManager.sendEvent(mouseRightPressed, mousePos);
	}

};
void eventProcess::_handleMouseButtonReleased(){
	static const Hash *mouseLeftReleased = Hash::getHash("mouseLeftReleasedScreen"); 
	static const Hash *mouseRightReleased = Hash::getHash("mouseRightReleasedScreen"); 

	vector2 mousePos = vector2(event.mouseButton.x, event.mouseButton.y);
	
	if(event.mouseButton.button == sf::Mouse::Button::Left){
		eventManager.sendEvent(mouseLeftReleased, mousePos);
	}
	else if(event.mouseButton.button == sf::Mouse::Button::Right){
		eventManager.sendEvent(mouseRightReleased, mousePos);
	}
};

void eventProcess::_handleMouseMove(){
	static const Hash *mouseMoved = Hash::getHash("mouseMovedScreen"); 

	vector2 mousePos = vector2(event.mouseMove.x, event.mouseMove.y);
	
	eventManager.sendEvent(mouseMoved, mousePos);

};

void eventProcess::_handleKeyboardPressed(){
	static const Hash *keyPressed = Hash::getHash("keyPressed"); 

	sf::Event::KeyEvent keyEvent = event.key;
	
	eventManager.sendEvent(keyPressed, keyEvent);
};
void eventProcess::_handleKeyboardReleased(){
	static const Hash *keyReleased = Hash::getHash("keyReleased"); 

	sf::Event::KeyEvent keyEvent = event.key;
	
	eventManager.sendEvent(keyReleased, keyEvent);
};


void eventProcess::_handleMouseWheelMove(){
	static const Hash *mouseWheelUp = Hash::getHash("mouseWheelUp"); 
	static const Hash *mouseWheelDown = Hash::getHash("mouseWheelDown"); 

	sf::Event::MouseWheelEvent mouseWheelEvent = event.mouseWheel;

	int delta = event.mouseWheel.delta;

	if(delta > 0){
		eventManager.sendEvent<int>(mouseWheelUp, delta);
	}else{
		delta = -delta;
		eventManager.sendEvent<int>(mouseWheelDown, delta);
	}
};


#pragma once
#include "Object.h"
#include "objectProcessor.h"

#include <vector>
#include "../util/logObject.h"


#pragma once
#include "../../core/State/State.h"
#include "../../core/State/dummyStateSaveLoader.h"
#include "../../core/objectMgr.h"
#include "../../core/Process/objectMgrProcess.h"
   
        

#include "../factory/objectFactory.h"
#include "../gameStateHelpers/playerEventHandler.h"
#include "../gameStateHelpers/gunsManager.h"
#include "../gameStateHelpers/playerController.h"

class gameState : public State{
public:
	gameState() : State("gameState"){};
	
	void Update(float dt){
		this->_playerController->Update(dt);
	};
	
	void Draw(){};

	stateSaveLoader *createSaveLoader(){
		return new dummyStateSaveLoader();
	}

protected:
	void _Init();

	void _initFactory();
	void _generateBoundary(vector2 levelDim);
	void _generateTerrain(unsigned long long seed, vector2 playerInitPos, vector2 levelDim);
	void _createPlayer(vector2 playerInitPos, vector2 levelDim);
	void _createEnemies(vector2 levelDim);
	void _createDummy(vector2 levelDim);
	
	
	Object* _createGuns(Object *player, vector2 levelDim);

	objectMgr *objectManager;
	viewProcess *viewProc;

	playerController *_playerController;

	objectFactory objFactory; 
};


#pragma once
#include <math.h>
#include <algorithm>

#ifndef PRINTVECTOR2
	#define PRINTVECTOR2(vec) std::cout<<"\n\t"<<#vec<<" X = "<<((vec).x)<<" Y = "<<((vec).y)<<std::endl;
#endif

//HACK!--------------------------------------
#include "../include/Box2D/Common/b2Math.h"
#include "../include/SFML/System/Vector2.hpp"
//class b2Vec2;


/*!represents a 2-dimensional vector*/
class vector2{
public:
	float x, y;
 	
 	/*!created a null vector*/
	vector2(){ this->x = this->y = 0;};
	~vector2(){};

	/*!typecast a vector of another type to a vector2
	
	the other vector *must* have x and y as public
	member variables. otherwise, this function will not work

	@param [in] otherVec the vector of another type
	\return a vector2 with the same x and y coordinate
	*/
	template<typename T>
	static vector2 cast(const T &otherVec){
		return vector2(otherVec.x, otherVec.y);
	}

	/*!typecasts a vector2 to a vector of another type

	the other vector *must* have a constructor of the form
	otherVector(float x, float y). Otherwise, this function will not work
	
	\return a vector of the other type
	*/
	template<typename T>
	T cast() const{
		return T(this->x, this->y);
	}

	/*!construct a vector
	@param [in] x the x-coordinate
	@param [in] y the y-coordinate
	*/
	/*inline*/ vector2(float x, float y){
		this->x = x; this->y = y;
	};

	inline vector2(const vector2& other){
		this->x = other.x;
		this->y = other.y;
	}
	

	/*!normalize the vector
	
	creates a new vector which has the same direction 
	as this vector, but has magnitude one

	\return a new unit vector in this vector's direction
	*/
	vector2 Normalize() const{	
		float length = this->Length(); 
		length  = (length == 0 ? 1 : length);
		return (vector2(this->x / length, this->y / length));
	};


	/*!return the angle made by this vector with the x axis in
	counter clockwise direction *in radians*
	*/
	float toAngle() const{
		return atan2(this->y, this->x);
	}

	/*!return a vector that is clamped between minVec and maxVec
	*/   
	vector2 clamp(vector2 minVec, vector2 maxVec){
		vector2 clampedVec; clampedVec.x = x; clampedVec.y = y;
		if(clampedVec.x < minVec.x) clampedVec.x = minVec.x;
		if(clampedVec.x > maxVec.x) clampedVec.x = maxVec.x;

		if(clampedVec.y < minVec.y) clampedVec.y = minVec.y;
		if(clampedVec.y > maxVec.y) clampedVec.y = maxVec.y;

		return clampedVec;
	};

	float dotProduct(vector2 other){
		return this->x * other.x + this->y * other.y;
	}
	/*!projects *this* vector onto the other vector*/
	vector2 projectOn(vector2 projectDir){

		vector2 normalizedProjectDir = projectDir.Normalize();
		//normalize the other vector and multiply it by *this* vector's
		//component in the other vector's direction;
		return normalizedProjectDir * (this->dotProduct(normalizedProjectDir));
	}

	/*!returns the length of the  vector*/
	inline float Length() const{ return (sqrt(x * x  +  y * y)); };
	/*!returns the length of the vector squared.
	For performance, use this instead of vector2::Length to compare distances.
	*/
	inline float LengthSquared() const{ return (x * x + y * y); };

	//---------operator overloads-------------------------------------------

	/*!negate this vector*/
	inline vector2 operator -(){	return vector2(-x, -y); };
	/*!Add a vector to this vector.*/
	inline void operator += (const vector2& v){ x += v.x; y += v.y; };
	/*!Subtract a vector from this vector.*/
	inline void operator -= (const vector2& v){ x -= v.x; y -= v.y; };
	/*!Multiply this vector by a scalar.*/
	inline void operator *= (float a){ x *= a; y *= a; };

	inline vector2 operator + (const vector2& a) const { return vector2(x + a.x, y + a.y); };
	inline vector2 operator - (const vector2& a) const { return vector2(x - a.x, y - a.y); };
	inline vector2 operator / (const vector2& a) const { return vector2(x / a.x, y / a.y); };
	inline vector2 operator * (float scale)		 const { return vector2(x * scale, y * scale); };

	inline bool operator > (const vector2& a) const  { return (this->x > a.x && this->y > a.y); };
	inline bool operator < (const vector2& a) const  { return (this->x < a.x && this->y < a.y); };
	inline bool operator >= (const vector2& a) const { return (this->x >= a.x && this->y >= a.y); };
	inline bool operator <= (const vector2& a) const { return (this->x <= a.x && this->y <= a.y); };
	inline bool operator == (const vector2& a) const { return (this->x == a.x && this->y == a.y); };
	inline bool operator != (const vector2& a) const { return (this->x != a.x || this->y != a.y); };
	
	inline operator b2Vec2(){ return b2Vec2(this->x, this->y); }
	
	template <typename T>
	inline operator sf::Vector2<T>(){ return sf::Vector2<T>(this->x, this->y); }

};

#define zeroVector (vector2(0, 0))
#define nullVector (vector2(0, 0))

template<typename TYPE>
inline vector2 operator * (const TYPE s, const vector2& a) { return vector2(a.x * s  , a.y * s);    };	

template<typename TYPE>
inline vector2 operator * (const vector2& a, const TYPE s) { return vector2(a.x * s  , a.y * s);    };	

inline bool    operator == (const vector2&a , vector2& b) { return (a.x == a.y) && (b.x == b.y);  };

//------------------------------------------------------------------------------------------------

/*!represents a 3-d vector*/
class vector3{
public:
	float x, y, z;

	/*!create a null vector*/
	vector3(){ this->x = this->y = this->z = 0;};

	/*!create a vector
	@param [in] x the x-coordinate
	@param [in] y the y-coordinate
	@param [in] z the z-coordinate
	*/
	vector3(float x, float y, float z){
		this->x = x; this->y = y;
		this->z = z;
	};

	/*!create a vector
	@param [in] vec2 the x and y coordinates 
	@param [in] z the z coordinate 
	*/
	vector3(vector2 vec2, float z){
		this->x = vec2.x;
		this->y = vec2.y;
		this->z = z;
	}

	/*!normalize the vector

	creates a new vector which has the same direction 
	as this vector, but has magnitude one

	\return a new unit vector in this vector's direction
	*/
	vector3 Normalize(){	
		float length = this->Length(); 
		length  = length == 0 ? 1 : length;
		return (vector3(this->x / length, this->y / length, this->z / length));
	};


	/*!returns the length of the  vector*/
	inline float Length(){ return (sqrt(x * x  +  y * y + z * z)); };
	/*!returns the length of the vector squared.
		For performance, use this instead of vector2::Length to compare distances.
		*/
	inline float LengthSquared() const{ return (x * x + y * y + z * z); };

	//---------operator overloads-------------------------------------------

	//negate this vector
	inline vector3 operator -(){	return vector3(-x, -y, -z); };
	// Add a vector3 to this vector.
	inline void operator += (const vector3& v){ x += v.x; y += v.y; z += v.z; };
	// Add a vector2 to this vector.
	inline void operator += (const vector2& v){ x += v.x; y += v.y; };

	// Subtract a vector3 from this vector.
	inline void operator -= (const vector3& v){ x -= v.x; y -= v.y; z -= v.z;};
	// Subtract a vector2 from this vector.
	inline void operator -= (const vector2& v){ x -= v.x; y -= v.y;};

	// Multiply this vector by a scalar.
	inline void operator *= (float a){ x *= a; y *= a; z *= a; };

	inline vector3 operator + (const vector3& a) { return vector3(x + a.x, y + a.y, z + a.z); };
	inline vector2 operator + (const vector2& a) { return vector2(x + a.x, y + a.y); };

	inline vector3 operator - (const vector3& a) { return vector3(x - a.x, y - a.y, z - a.z); };
	inline vector2 operator - (const vector2& a) { return vector2(x - a.x, y - a.y); };


	inline bool operator > (const vector3& a) { return (this->x > a.x && this->y > a.y && this->z > a.z); };
	inline bool operator < (const vector3& a) { return (this->x < a.x && this->y < a.y && this->z < a.z); };
	inline bool operator >= (const vector3& a) { return (this->x >= a.x && this->y >= a.y && this->z >= a.z); };
	inline bool operator <= (const vector3& a) { return (this->x <= a.x && this->y <= a.y  && this->z <= a.z); };

	//templatized function to typecast an object to another one by using the constructor -_^
	
	//template<class otherPhyVect> inline operator otherPhyVect(){ return otherPhyVect(this->x, this->y, this->z); }
	template<class otherPhyVect2> inline operator otherPhyVect2(){ return otherPhyVect2(this->x, this->y); }
	
	//conversion from vector3 to vector2
	inline operator vector2(){ return vector2(this->x, this->y); }

};
inline vector3 operator * (float s, const vector3& a)		   { return vector3(a.x * s  , a.y * s, a.z * s);    };	
inline vector3 operator * (const vector3& a, float s)		   { return vector3(a.x * s  , a.y * s, a.z * s);	 };	
inline bool    operator == (const vector3&a , vector3& b)	   { return (a.x == a.y) && (b.x == b.y) && (a.z == b.z);  };


const char *token_names[] =
{
   [CT_POUND]        = "POUND",
   [CT_PREPROC]      = "PREPROC",
   [CT_PREPROC_BODY] = "PREPROC_BODY",
   [CT_PP]           = "PP",
};


int main(int argc, char *argv[])
{
   struct junk a[] =
   {
      { "version",  0,   0,   0 },
      { "file",     1, 150, 'f' },
      { "config",   1,   0, 'c' },
      { "parsed",  25,   0, 'p' },
      { NULL,       0,   0,   0 }
   };
}


color_t colors[] =
{
   { "red",   { 255, 0,   0 } }, { "blue",   {   0, 255, 0 } },
   { "green", {   0, 0, 255 } }, { "purple", { 255, 255, 0 } },
};


struct foo_t bar =
{
   .name = "bar",
   .age  = 21
};


struct foo_t bars[] =
{
   [0] = { .name = "bar",
           .age  = 21 },
   [1] = { .name = "barley",
           .age  = 55 },
};

void foo(void)
{
   int  i;
   char *name;

   i    = 5;
   name = "bob";
}

/**
 * This is your typical header comment
 */
int foo(int bar)
{
   int idx;
   int res = 0;      // trailing comment
                     // that spans two lines
   for (idx = 1; idx < bar; idx++)
      /* comment in virtual braces */
      res += idx;

   res *= idx;        // some comment

   // almost continued, but a NL in between

// col1 comment in level 1
   return(res);
}

// col1 comment in level 0


#define foobar(x)             \
   {                          \
      for (i = 0; i < x; i++) \
      {                       \
         junk(i, x);          \
      }                       \
   }


void foo(void)
{
   switch(ch)
   {
   case 'a':
      {
         handle_a();
         break;
      }

   case 'b':
      handle_b();
      break;

   case 'c':
   case 'd':
      handle_cd();
      break;

   case 'e':
      {
         handle_a();
      }
      break;

   default:
      handle_default();
      break;
   }
}



//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------
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
	
	objectMap objMap;

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
	void addObject(Object *obj){

		assert(obj != NULL);
		util::infoLog<<"\n"<<obj->getName()<<"  added to objectManager";

		assert(this->objMap.find(obj->getName()) == this->objMap.end());

		this->objMap[obj->getName()] = obj;
		for(objProcessorIt it = this->objProcessors.begin(); it != this->objProcessors.end(); ++it){

			
			(*it)->onObjectAdd(obj);
		}

	}
	
	/*!remove an Object from the objectManager
	this is rarely, if every used. Once an object is removed,
	it will not be processed by objectProcessors
	
	@param [in] name the unique name of the Object
	*/
	void removeObject(std::string name){
		objMapIt it = this->objMap.find(name);

		if(it != this->objMap.end()){
			this->objMap.erase(it);
			Object *obj = it->second;

			for(objectProcessor* processor : objProcessors){
				processor->onObjectRemove(obj);
			}

		}
	}

	void removeObject(Object &obj){
		this->removeObject(obj.getName());
	}

	void removeObject(Object *obj){
		assert(obj != NULL);

		this->removeObject(obj->getName());
	}

	/*!returns an Object by it's unique name
	@param [in] name the unique name of the Object

	\return the Object with the given name, or NULL
	if the Object does not exist
	*/
	Object *getObjByName(std::string name){
		objMapIt it = this->objMap.find(name);

		if(  it == this->objMap.end() ){
			return NULL;
		}
		return it->second;
		
	}

	/*!add an objectProcessor to make it process Objects
	@param [in] processor the objectProcessor to be added
	*/
	void addObjectProcessor(objectProcessor *processor){
		this->objProcessors.push_back(processor);
		processor->Init(&this->objMap);
	}

	/*!remove an objectProcessor that had been added
	@param [in] processor the objectProcessor to be removed 
	*/
	void removeObjectProcessor(objectProcessor *processor){
		//this->objProcessors.push_back(processor);
	}

	/*!pre-processing takes place
	generally, variables are setup if need be in this step, 
	and everything is made ready for Process
	*/
	void preProcess(){
		for(objectProcessor* processor : objProcessors){
			processor->preProcess();
		}
	}

	/*!cleaning up takes place
	once Process is called, postProcess takes care of cleaning
	behind all of the leftover variables and data.
	Dead Objects are also destroyed in this step
	*/
	void postProcess(){
		for(objectProcessor* processor : objProcessors){
			processor->postProcess();
		}

		
		for(auto it = this->objMap.begin(); it != this->objMap.end(); ){
			Object *obj = it->second;

			if(obj->isDead()){
				it = objMap.erase(it);

				for(objectProcessor* processor : objProcessors){
					processor->onObjectRemove(obj);
				}

				delete(obj);
				obj = NULL;
			}else{
				it++;
			}
		}
		
	}

	/*!Processes all objects*/
	void Process(float dt){
		
		for(objectProcessor* processor : objProcessors){
			processor->Process(dt);
		}
	};
};

//GAME STATE-------------------------------------------------------------------------------
#pragma once
#include "gameState.h"
#include "../../core/Object.h"
#include "../../core/renderUtil.h"

#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/cameraProcessor.h"



void gameState::_Init(){

	objectMgrProcess *objMgrProc = this->processManager->getProcess<objectMgrProcess>(
		Hash::getHash("objectMgrProcess"));
	
	this->objectManager = objMgrProc->getObjectMgr();

	this->viewProc = this->processManager->getProcess<viewProcess>(
		Hash::getHash("viewProcess"));
	
	this->_initFactory();

	vector2 playerInitPos = viewProc->view2GameCoord(vector2(300, 300));
	vector2 levelDim = viewProc->view2GameCoord(vector2(3000, 2000));
	
	
	
	this->_generateBoundary(levelDim);
	this->_generateTerrain(0, playerInitPos, levelDim);
	this->_createEnemies(levelDim);
	this->_createDummy(levelDim);
	this->_createPlayer(playerInitPos, levelDim);
	
}


//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------


#include "../factory/playerCreator.h"
#include "../factory/boundaryCreator.h"
#include "../factory/dummyCreator.h"
#include "../factory/bulletCreator.h"
#include "../factory/gunCreator.h"
#include "../factory/terrainCreator.h"
#include "../factory/pickupCreator.h"
#include "../factory/bladeCreator.h"

void gameState::_initFactory(){

	
	this->objFactory.attachObjectCreator(Hash::getHash("dummy"),
				new dummyCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("player"),
				new playerCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("boundary"),
				new boundaryCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("bullet"),
				new bulletCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("gun"),
					new gunCreator(this->viewProc));
	this->objFactory.attachObjectCreator(Hash::getHash("terrain"),
					new terrainCreator(this->viewProc));
	this->objFactory.attachObjectCreator(Hash::getHash("pickup"),
					new pickupCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("blade"),
					new bladeCreator(this->viewProc));
};


void gameState::_generateTerrain(unsigned long long seed, 
	vector2 playerInitPos, vector2 levelDim){

	terrainCreator *creator = objFactory.getCreator<terrainCreator>(
		Hash::getHash("terrain"));

 	vector2 blockDim =  vector2(64, 64);
 	vector2 terrainDim =  vector2(levelDim.x / blockDim.x, 
 		levelDim.y / blockDim.y); 
 	vector2 minPos   =  vector2(0, 0);
 	vector2 maxPos = minPos + vector2(blockDim.x * terrainDim.x, 
 					blockDim.y * terrainDim.y);
 	float render2GameCoord =  viewProc->getRender2GameScale();



 	minPos *= render2GameCoord;
 	blockDim *= render2GameCoord;
 	maxPos *= render2GameCoord;

	creator->setBounds(minPos, maxPos, blockDim);
	creator->reserveRectSpace(playerInitPos, 
		vector2(256, 256) * render2GameCoord);


	
	Object *terrainObj = creator->createObject();


};



void gameState::_createPlayer(vector2 playerInitPos, vector2 levelDim){

	playerCreator *creator = objFactory.getCreator<playerCreator>(
		Hash::getHash("player"));

	//players handlers--------------------------------------------------
	playerHandlerData playerData;
	playerData.left = sf::Keyboard::Key::A;
	playerData.right = sf::Keyboard::Key::D;
	playerData.up = sf::Keyboard::Key::W;
	playerData.fireGun = sf::Keyboard::Key::S;



	this->_playerController = new playerController(this->eventManager, this->objectManager,
						&this->objFactory, this->viewProc);

	this->_playerController->createPlayer(levelDim, playerInitPos, creator,
				playerData);


	{
	bladeData blade;

	bladeCreator *creator = this->objFactory.getCreator<bladeCreator>(Hash::getHash("blade"));
	creator->setParent(this->_playerController->getPlayer());

	Object *obj = creator->createObject(vector2(300, 300));

	objectManager->addObject(obj);
	}

	
};


#include "../generators/gunDataGenerator.h"
void gameState::_createDummy(vector2 levelDim){
	{

		dummyCreator *creator = objFactory.getCreator<dummyCreator>(
			Hash::getHash("dummy"));

		
		creator->setRadius(1.0f);

		vector2 randPos = vector2(400, 200);
		
		randPos *= viewProc->getRender2GameScale();
		Object *dummy = creator->createObject(randPos);
		objectManager->addObject(dummy);
	}

	

	{	
	
		pickupCreator *creator = objFactory.getCreator<pickupCreator>(
			Hash::getHash("pickup"));

		creator->setCollisionRadius(1.0f);

		pickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<gunDataGenerator>(
			gunDataGenerator(gunDataGenerator::Archetype::Rocket, 
				1, 10));
		
		
	
		creator->setPickupData(data);
		vector2 pos = vector2(600, 100);
		pos *= viewProc->getRender2GameScale();
		Object *obj = creator->createObject(pos);
		objectManager->addObject(obj);


	}


	{	
	
		pickupCreator *creator = objFactory.getCreator<pickupCreator>(
			Hash::getHash("pickup"));

		creator->setCollisionRadius(1.0f);

		pickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<gunDataGenerator>(
			gunDataGenerator(gunDataGenerator::Archetype::machineGun, 
				1, 10));
		
		
	
		creator->setPickupData(data);


		vector2 pos = vector2(1200, 100);
		pos *= viewProc->getRender2GameScale();


		Object *obj = creator->createObject(pos);
		objectManager->addObject(obj);


	}
};


#include "../bulletColliders/pushCollider.h"
#include "../bulletColliders/damageCollider.h"
Object* gameState::_createGuns(Object *player, vector2 levelDim){
	bulletCreator *_bulletCreator = objFactory.getCreator<bulletCreator>(
		Hash::getHash("bullet"));

	gunCreator *creator = objFactory.getCreator<gunCreator>(
		Hash::getHash("gun"));

	bulletData bullet;
	bullet.addEnemyCollision(Hash::getHash("enemy"));
	bullet.addEnemyCollision(Hash::getHash("dummy"));
	bullet.addIgnoreCollision(Hash::getHash("player"));
	bullet.addIgnoreCollision(Hash::getHash("pickup"));

	bullet.addBulletCollder(new pushCollider(2.0));
	bullet.addBulletCollder(new damageCollider(1.0));

	gunData data;
	data.setClipSize(100);
	data.setClipCooldown(100);
	data.setShotCooldown(3);
	data.setBulletRadius(0.3);
	data.setBulletCreator(_bulletCreator);
	data.setBulletData(bullet);
	data.setBulletVel(40);

	vector2 pos = vector2(400, 200);
	pos *= viewProc->getRender2GameScale();

	creator->setGunData(data);
	creator->setParent(player);


	Object *gun = creator->createObject(pos);
	objectManager->addObject(gun);

	return gun;
};


void gameState::_generateBoundary(vector2 levelDim){

	boundaryCreator *creator = objFactory.getCreator<boundaryCreator>(
		Hash::getHash("boundary"));

	creator->setBoundaryThickness(3.0f);
	creator->setDimensions(levelDim);

	Object *boundary = creator->createObject(vector2(0, -200));


	objectManager->addObject(boundary);

}

void gameState::_createEnemies(vector2 levelDim){

	/*

	bulletCreator *creator = (bulletCreator *)objFactory.getCreator(
		Hash::getHash("bullet"));


	bulletData data;
	data.addEnemyCollision(Hash::getHash("dummy"));


	creator->setBulletData(data);
	creator->setCollisionRadius(0.8f);

	vector2 pos = vector2(400, 400);
	pos *= viewProc->getRender2GameScale();

	Object *obj = creator->createObject(pos);
	

	objectManager->addObject(obj);
	
	*/
};

//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------

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

		
		/*! constructor used to create a msgLog

		If the logLevel is greater than the logThreshold set in baseLog,
		then the message will be emitted. 

		If a logLevel of Error is used, and Error is above the log threshold,
		then the error will be emitted and the program will halt 
		
		@param [in] msg the message to be printed
		@param [in] level the logLevel of the given message.
		
		\sa scopedLog
		*/
		/*
		msgLog(std::string msg){
			if(level >= baseLog::thresholdLevel){
				std::cout<<"\n"<<msg<<std::endl;

				if(level == logLevelError){
					std::cout<<"\n\n Quitting from logObject due to error"<<std::endl;
					assert(false);
				}
			}

			
		};*/

		template <typename T>
		msgLog & operator << (T toWrite){
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



//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------

#pragma once
#include "objContactListener.h"
#include "phyProcessor.h"


void objContactListener::_extractPhyData(b2Contact *contact, Object **a, Object **b){
	b2Fixture*fixtureA = contact->GetFixtureA();
	b2Fixture*fixtureB = contact->GetFixtureB();

	b2Body*bodyA = fixtureA->GetBody();
	b2Body*bodyB = fixtureB->GetBody();


	*a = static_cast<Object *>(bodyA->GetUserData());
	*b = static_cast<Object *>(bodyB->GetUserData());

	assert(*a != NULL && *b != NULL);
}

void objContactListener::BeginContact(b2Contact*contact){
	
	this->_handleCollision(collisionData::Type::onBegin,contact);

};
void objContactListener::EndContact(b2Contact*contact){

	this->_handleCollision(collisionData::Type::onEnd,contact);
	
};




collisionData objContactListener::_fillCollisionData(b2Contact *contact,
  Object *me,				 
  Object *otherPhy 		, 
phyData *myPhy, phyData *		otherPhy){

	collisionData collision;

	collision.myPhy = myPhy;
	collision.otherPhy = otherPhy;
	collision.otherObj = other;

	collision.myApproachVel = vector2::cast(myPhy->body->GetLinearVelocity());

	b2Manifold *localManifold = contact->GetManifold();
	b2WorldManifold worldManifold;

	 contact->GetWorldManifold(&worldManifold);
	//if(collision.normal == zeroVector){

	/*1) if the point exists, use it.
	the point is the most accurate way of figuring out the collision
	  
	2)if no point, use the collision normal  
	  this is *NOT* the actual collision normal. it is the vector that points
	  in the direction such that the two bodies can be seperated with the most 
	  ease in this direction.

	  3) if neither, make a sucky ballpark estimation based to relative velocities 

	  */
	if(vector2::cast(worldManifold.normal) != zeroVector){
		collision.normal = vector2::cast(worldManifold.normal);
	}
	else if(vector2::cast(localManifold->localNormal) != zeroVector){
	
		collision.normal = vector2::cast(localManifold->localNormal);
	}
	else if(vector2::cast(localManifold->localPoint) != zeroVector){
		collision.normal = vector2::cast(localManifold->localPoint);
	}
	
	 else if(localManifold->pointCount > 0){
		collision.normal = vector2::cast(localManifold->localPoint);
	} 
	else{
		collision.normal = nullVector;
	}
	

	collision.normal.Normalize();
	 
	return collision;
	
};	



void objContactListener::_handleCollision(collisionData::Type type, b2Contact *contact){
	
	collisionData collision;
	Object *a, *b;
	
	this->_extractPhyData(contact, &a, &b);
	
	assert(a != NULL && b != NULL);

	phyData *aPhyData = a->getProp<phyData>(Hash::getHash("phyData"));
	phyData *bPhyData = b->getProp<phyData>(Hash::getHash("phyData"));
	assert(aPhyData != NULL && bPhyData != NULL);

	/* a to b */
	collision = this->_fillCollisionData(contact, a, b, aPhyData, bPhyData);
	collision.type = type;

	aPhyData->addCollision(collision);

	/*b to a*/
	collision = this->_fillCollisionData(contact, b, a, bPhyData, aPhyData);
	collision.type = type;
	bPhyData->addCollision(collision);

	switch(a) { 
case 1: { return 0;}
			case 2: int i = 0; break; case 3: {
			for(int i = 0; ; ){ std::cout<<i; }
			return NULL;}

		case 4: { int k = 1; } break;
		}
			
		
};	


