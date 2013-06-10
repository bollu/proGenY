#pragma once
#include <iostream>
/*
#include "include/Box2D/Box2D.h"
#include "include/SFML/Graphics.hpp"

#include "core/Object.h"
#include "core/objectMgr.h"
#include "core/ObjProcessors/renderProcessor.h"
#include "core/ObjProcessors/phyProcessor.h"

#include "util/logObject.h" 
*/
//TESTING INCLUDES----------------------------------------
#include "core/renderUtil.h"


//PRODUCTION INCLUDEs-----------------------------------------
#include "include/SFML/System/Clock.hpp"
#include "core/Messaging/eventMgr.h"

#include "core/Process/Process.h"
#include "core/Process/processMgr.h"

//processes---------------------------------------
#include "core/Process/windowProcess.h"
#include "core/Process/eventProcess.h"
#include "core/Process/objectMgrProcess.h"
#include "core/Process/worldProcess.h"
#include "core/Process/viewProcess.h"
#include "core/Process/stateProcess.h"


//GAME STATES----------------------------------------------------
#include "game/States/mainMenuState.h"
#include "game/States/gameState.h"


void _loadSettings(Settings &);
void _addProcesses(processMgr &, Settings &, eventMgr &);
void _createObjectProcessors(objectMgrProcess*, processMgr&, Settings &, eventMgr&);
void _createStates(stateProcess *);




void _createDummy(objectMgr *);

int main(){
    processMgr processManager;
    Settings settings;
    eventMgr eventManager;
    sf::Clock Clock;


    _loadSettings(settings);
    //settings.loadSettingsFromFile(".settings.ini");

    _addProcesses(processManager, settings, eventManager);


    bool done = false;
    float dt = 0.0;

    while(!done){
       sf::Time elapsed = Clock.restart();
       dt = elapsed.asSeconds();


       processManager.preUpdate();
       processManager.Update(dt);
       processManager.Draw();
       processManager.postDraw();

      // sf::sleep( sf::milliseconds(rand() % 30) );
   }

   processManager.Shutdown();

   return 0;
}


void _loadSettings(Settings &settings){
    //TODO: actually load settings here. for now, just create the settings
    //and just load it back.

    settings.addProp(Hash::getHash("screenDimensions"), new v2Prop(vector2(1280, 720)));

    settings.addProp(Hash::getHash("gravity"), new v2Prop(vector2(0, -15.0)));
     settings.addProp(Hash::getHash("stepSize"), new fProp(1.0f / 60.0f));
    settings.addProp(Hash::getHash("velIterations"), new iProp(5));
    settings.addProp(Hash::getHash("collisionIterations"), new iProp(5));

    

};


void _addProcesses(processMgr &processManager, Settings &settings, eventMgr &eventManager){


    //DO NOT CHANGE THE ORDER. SOME COMPONENTS DEPEND ON OTHERS
    processManager.addProcess(new worldProcess(processManager, settings, eventManager));
    processManager.addProcess(new windowProcess(processManager, settings, eventManager));


    processManager.addProcess(new eventProcess(processManager, settings, eventManager));
    processManager.addProcess(new viewProcess(processManager, settings, eventManager));

    //ALWAYS KEEP THIS LAST BUT ONE.It will depend on most other components
    //But other game states will probably rely on this.
    objectMgrProcess *objMgrProc = new objectMgrProcess(processManager, settings, eventManager);

    //create the object processors that are responsible for creating objects
    _createObjectProcessors(objMgrProc, processManager, settings, eventManager);
    processManager.addProcess(objMgrProc);


     //KEEP THIS ONE THE LAST ONE> it depends on all other processes.
    stateProcess *stateProc = new stateProcess(processManager, settings, eventManager);
     //create the game states
    _createStates(stateProc);
     //add the stateProcess
    processManager.addProcess(stateProc);
};



void _createStates(stateProcess *stateProc){
    stateProc->addState(new mainMenuState(), false);
    stateProc->addState(new gameState(), true);
}

//OBJECT PROCESSORS------------------------------------------------------------
#include "game/ObjProcessors/terrainProcessor.h"
#include "game/ObjProcessors/cameraProcessor.h"
#include "game/ObjProcessors/bulletProcessor.h"
#include "game/ObjProcessors/healthProcessor.h"

void _createObjectProcessors(objectMgrProcess *objMgrProc, processMgr &processManager,
                           Settings &settings, eventMgr &eventManager){


    objMgrProc->addObjectProcessor( new terrainProcessor(processManager, settings, eventManager));
    objMgrProc->addObjectProcessor( new cameraProcessor(processManager, settings, eventManager));
     
     
    objMgrProc->addObjectProcessor( new renderProcessor(processManager, settings, eventManager));
    objMgrProc->addObjectProcessor( new phyProcessor(processManager, settings, eventManager));

    objMgrProc->addObjectProcessor(new groundMoveProcessor(processManager, settings, eventManager) );
    objMgrProc->addObjectProcessor(new bulletProcessor(processManager, settings, eventManager) );
    objMgrProc->addObjectProcessor(new healthProcessor(processManager, settings, eventManager) );

};




/*
//global objects---------------------------------------------------------
//scaling factor b/w box2d and SFML render Coordinates
// 1 box2d unit = 20 rendering units
static const float b2ToRenderScale = 20.0;

b2World world(b2Vec2(0, -9.8 ));
sf::RenderWindow window(sf::VideoMode(1280, 720), "SFML asd!");
sf::View camera(sf::FloatRect(0, 0, 1280, 720));

objectMgr objManager;

//function-------------------------------------------------------------
void _Init();
void _handleEvents(sf::Event &event);
void _Update();
void _Draw();

//main---------------------------------------------------------------------
int main(){
    _Init();

    objManager.addObjectProcessor( new renderProcessor(window));
    objManager.addObjectProcessor( new phyProcessor(world));

    
    
    Object *dummy = new Object("dummy"); 

   
    //dummyShape->SetPosition(vector2(100, 100));

    b2BodyDef dummyBodyDef;
    dummyBodyDef.type = b2_dynamicBody;

    b2PolygonShape groundBox; 
    groundBox.SetAsBox(0.2f, 0.2f);

    b2FixtureDef dummyFixtureDef;
    dummyFixtureDef.shape = &groundBox;
    dummyFixtureDef.friction = 1.0;

    sf::Shape *dummyShape = renderUtil::createShape(&groundBox);


    dummyShape->setFillColor(sf::Color::Red);
    dummyShape->setOutlineColor(sf::Color::Red);
    dummyShape->setOutlineThickness(5);


    dummy->addProp(Hash::getHash("shapeDrawable"), 
        new managedProp<sf::Shape>(dummyShape) );

    dummy->getProp<vector2>(Hash::getHash("position"))->setVal(vector2(640, 380));
 

    dummy->addProp(Hash::getHash("b2BodyDef"), new Prop<b2BodyDef>(dummyBodyDef));
    dummy->addProp(Hash::getHash("b2FixtureDef"), new Prop<b2FixtureDef>(dummyFixtureDef));

 objManager.addObject(dummy);
    

   




    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            _handleEvents(event);
        }

        _Update();
        _Draw();
    }

    return 0;
}


void _Init(){
     camera.zoom(1.0 / b2ToRenderScale);
     camera.setRotation(180); 
     window.setView(camera);
  
}


void _handleEvents(sf::Event &event){
  if (event.type == sf::Event::Closed){
    window.close();
}
};

void _Update(){
    world.Step(1.0 / 60.0, 5, 5);
    window.clear();
    objManager.Process(); 
}

void _Draw(){
    window.display();
}*/