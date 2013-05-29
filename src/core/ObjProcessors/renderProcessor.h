#pragma once
#include "../objectProcessor.h"
#include "../include/SFML/Graphics.hpp"
#include "../vector.h"


class renderProcessor : public objectProcessor{
private:
	sf::RenderWindow &window;
public:
	
	renderProcessor(sf::RenderWindow &_window) : window(_window){};


	void onObjectAdd(Object *obj){
		
	}

	void Process(){
		
		for(cObjMapIt it= this->objMap->begin(); it != this->objMap->end(); ++it){
			Object *obj = it->second;

			auto posProp = obj->getProp<vector2>(Hash::getHash("position"));

			//shape Drawable
			{
				auto drawableProp = obj->getManagedProp<sf::Shape>(Hash::getHash("shapeDrawable"));
				if(drawableProp != NULL){
					drawableProp->getVal()->setPosition(posProp->getVal());
					window.draw(*drawableProp->getVal());

					return;
				} 
			}

			//sprite Drawable
			{
				auto drawableProp = obj->getManagedProp<sf::Sprite>(Hash::getHash("spriteDrawable"));
				if(drawableProp != NULL){
					drawableProp->getVal()->setPosition(posProp->getVal());
					window.draw(*drawableProp->getVal());

					return;
				} 
			}

			//text Drawable
			{
				auto drawableProp = obj->getManagedProp<sf::Text>(Hash::getHash("textDrawable"));
				if(drawableProp != NULL){
					drawableProp->getVal()->setPosition(posProp->getVal());
					window.draw(*drawableProp->getVal());

					return;
				} 
			}


		};
	}
};
