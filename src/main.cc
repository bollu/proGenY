#pragma once
#include <iostream>
#include <thread>

/*#define BACKWARD_HAS_DW 1
#include "include/backward/backward.cpp"
//#include "include/backward/backward.hpp"
*/
/*
#include "include/Box2D/Box2D.h"
#include "include/SFML/Graphics.hpp"

#include "core/Object.h"
#include "core/ObjectMgr.h"
#include "core/ObjProcessors/RenderProcessor.h"
#include "core/ObjProcessors/PhyProcessor.h"

#include "util/logObject.h" 
*/


//PRODUCTION INCLUDEs-----------------------------------------
#include "include/SFML/System/Clock.hpp"
#include "core/controlFlow/eventMgr.h"

#include "core/controlFlow/Process.h"
#include "core/controlFlow/processMgr.h"

//processes---------------------------------------
#include "core/Rendering/windowProcess.h"
#include "core/IO/eventProcess.h"
#include "core/componentSys/ObjectMgrProcess.h"
#include "core/World/worldProcess.h"
#include "core/Rendering/viewProcess.h"
#include "core/controlFlow/stateProcess.h"
#include "core/Rendering/renderProcess.h"

//GAME STATES----------------------------------------------------
#include "game/States/mainMenuState.h"
#include "game/States/gameState.h"
#include "game/States/gameSegmentLoader.h"

//listeners-------------------------------------------------------
#include "mainLoopListener.h"

void _loadSettings(Settings &);
void _addProcesses(processMgr &, Settings &, eventMgr &);
void _createObjectProcessors(ObjectMgrProcess*, processMgr&, Settings &, eventMgr&);
void _createStates(stateProcess *);




void _createDummy(ObjectMgr *);


int main(){
    
    processMgr processManager;
    Settings settings;
    eventMgr eventManager;
    sf::Clock Clock;


    _loadSettings(settings);
    //settings.loadSettingsFromFile(".settings.ini");

    _addProcesses(processManager, settings, eventManager);


    mainLoopListener listener(eventManager);


    bool done = false;
    float dt = 0.0;

    while(!listener.isWindowClosed()){
       sf::Time elapsed = Clock.restart();
       dt = elapsed.asSeconds();


       processManager.preUpdate();

       processManager.Update(dt);
       processManager.Draw();
       processManager.postDraw();

       //sf::sleep( sf::milliseconds(rand() % 30) );
   }

   processManager.Shutdown();

   return 0;
}


void _loadSettings(Settings &settings){
    //TODO: actually load settings here. for now, just create the settings
    //and just load it back.

    settings.addProp(Hash::getHash("screenDimensions"), new v2Prop(vector2(1280, 720)));

    settings.addProp(Hash::getHash("gravity"), new v2Prop(vector2(0, -50.0)));
    settings.addProp(Hash::getHash("stepSize"), new fProp(1.0f / 60.0f));
    settings.addProp(Hash::getHash("velIterations"), new iProp(3));
    settings.addProp(Hash::getHash("collisionIterations"), new iProp(3));

    IO::baseLog::setThreshold(IO::logLevel::logLevelInfo);
};


void _addProcesses(processMgr &processManager, Settings &settings, eventMgr &eventManager){


    //DO NOT CHANGE THE ORDER. SOME COMPONENTS DEPEND ON OTHERS
    processManager.addProcess(new worldProcess(processManager, settings, eventManager));
    processManager.addProcess(new windowProcess(processManager, settings, eventManager));


    processManager.addProcess(new eventProcess(processManager, settings, eventManager));
    processManager.addProcess(new viewProcess(processManager, settings, eventManager));
    processManager.addProcess(new renderProcess(processManager, settings, eventManager));
    //ALWAYS KEEP THIS LAST BUT ONE.It will depend on most other components
    //But other game states will probably rely on this.
    ObjectMgrProcess *objMgrProc = new ObjectMgrProcess(processManager, settings, eventManager);
    processManager.addProcess(objMgrProc);
    
    //create the object processors that are responsible for creating objects
    _createObjectProcessors(objMgrProc, processManager, settings, eventManager);


     //KEEP THIS ONE THE LAST ONE. it depends on all other processes.
    stateProcess *stateProc = new stateProcess(processManager, settings, eventManager);
     //create the game states
    _createStates(stateProc);
     //add the stateProcess
    processManager.addProcess(stateProc);
};



void _createStates(stateProcess *stateProc){
    stateProc->addState(new mainMenuState(), false);
    stateProc->addState(new gameSegmentLoader(), false);
    stateProc->addState(new gameState(), true);
}

//OBJECT PROCESSORS------------------------------------------------------------
#include "game/ObjProcessors/TerrainProcessor.h"
#include "game/ObjProcessors/CameraProcessor.h"
#include "game/ObjProcessors/BulletProcessor.h"
#include "game/ObjProcessors/HealthProcessor.h"
#include "game/ObjProcessors/GunProcessor.h"
#include "game/ObjProcessors/OffsetProcessor.h"
#include "game/ObjProcessors/PickupProcessor.h"
#include "game/ObjProcessors/AIProcessor.h"
void _createObjectProcessors(ObjectMgrProcess *objMgrProc, processMgr &processManager,
                           Settings &settings, eventMgr &eventManager){


    objMgrProc->addObjectProcessor( new terrainProcessor(processManager, settings, eventManager));
    objMgrProc->addObjectProcessor( new cameraProcessor(processManager, settings, eventManager));
     
     
    objMgrProc->addObjectProcessor( new RenderProcessor(processManager, settings, eventManager));
    objMgrProc->addObjectProcessor( new PhyProcessor(processManager, settings, eventManager));

    objMgrProc->addObjectProcessor(new groundMoveProcessor(processManager, settings, eventManager) );
    objMgrProc->addObjectProcessor(new BulletProcessor(processManager, settings, eventManager) );
    objMgrProc->addObjectProcessor(new healthProcessor(processManager, settings, eventManager) );
    objMgrProc->addObjectProcessor(new GunProcessor(processManager, settings, eventManager) );
    objMgrProc->addObjectProcessor(new OffsetProcessor(processManager, settings, eventManager) );
    objMgrProc->addObjectProcessor(new PickupProcessor(processManager, settings, eventManager) );
    objMgrProc->addObjectProcessor(new AIProcessor(processManager, settings, eventManager) );

};
