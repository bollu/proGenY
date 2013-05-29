#pragma once
#include <iostream>

#include "include/Box2D/Box2D.h"
#include "include/SFML/Graphics.hpp"

#include "core/Object.h"
#include "core/objectMgr.h"
#include "core/ObjProcessors/renderProcessor.h"
#include "core/ObjProcessors/phyProcessor.h"
#include "core/renderUtil.h"
#include "util/logObject.h" 

//global objects---------------------------------------------------------
//scaling factor b/w box2d and SFML render Coordinates
// 1 box2d unit = 20 rendering units
static const float b2ToRenderScale = 20.0;

b2World world(b2Vec2(0, -9.8));
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
    groundBox.SetAsBox(2.5f, 0.5f);

    b2FixtureDef dummyFixtureDef;
    dummyFixtureDef.shape = &groundBox;
    dummyFixtureDef.friction = 1.0;

    sf::Shape *dummyShape = renderUtil::createShape(&groundBox);


    dummyShape->setFillColor(sf::Color::Red);
    dummyShape->setOutlineColor(sf::Color::Red);
    dummyShape->setOutlineThickness(5);


    dummy->addProperty(Hash::getHash("shapeDrawable"), 
        new managedProp<sf::Shape>(dummyShape) );

    dummy->addProperty(Hash::getHash("position"), 
        new v2Prop(vector2(640, 360)) ) ;

    dummy->addProperty(Hash::getHash("b2BodyDef"), new Prop<b2BodyDef>(dummyBodyDef));
    dummy->addProperty(Hash::getHash("b2FixtureDef"), new Prop<b2FixtureDef>(dummyFixtureDef));

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
}