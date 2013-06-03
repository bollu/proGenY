#pragma once
#include "renderProcessor.h"

Renderer::Renderer(sf::Shape *shape){
	this->data.shape = shape;
	this->type = Renderer::Type::Shape;
}


Renderer::Renderer(sf::Sprite *sprite){
	this->data.sprite = sprite;
	this->type = Renderer::Type::Sprite;
}

Renderer::Renderer(sf::Text *text){
	this->data.text = text;
	this->type = Renderer::Type::Text;
}

void Renderer::deleteData(){
	if(this->type == Renderer::Type::Shape){
		delete(this->data.shape);
	}
	else if(this->type == Renderer::Type::Text){
		delete(this->data.text);
	}

	else if(this->type == Renderer::Type::Sprite){
		delete(this->data.sprite);
	}
}


Renderer::Renderer(const Renderer &other){
	this->type = other.type;

	if(this->type == Renderer::Type::Sprite){
		this->data.sprite = other.data.sprite;
	}
	if(this->type == Renderer::Type::Shape){
		this->data.shape = other.data.shape;
	}
	if(this->type == Renderer::Type::Text){
		this->data.text = other.data.text;
	}
};


renderProcessor::renderProcessor(sf::RenderWindow &_window, viewProcess &_view) : 
	window(_window), view(_view){};


void renderProcessor::onObjectAdd(Object *obj){
}

void renderProcessor::Process(float dt){

	for(auto it= objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;

		vector2* pos = obj->getProp<vector2>(Hash::getHash("position"));
		vector2 renderPos = view.game2ScreenCoord(*pos);

		renderData *data = obj->getProp<renderData>(Hash::getHash("renderData"));

		if(data == NULL){
			continue;
		}

		this->_Render(renderPos, data);
	};
}


void renderProcessor::_Render(vector2 pos, renderData *data){
		//loop through the renderers
	for(auto it = data->renderers.begin(); it != data->renderers.end(); ++it){
		Renderer &renderer = *it;


		switch(renderer.type){
			case Renderer::Type::Sprite:
			{
				sf::Sprite *sprite = renderer.data.sprite;
				sprite->setPosition(pos);
				window.draw(*sprite);
			
			}
			break;

			case Renderer::Type::Shape:
			{
				sf::Shape *shape = renderer.data.shape;
				shape->setPosition(pos);
				window.draw(*shape);
			}
			break;

			case Renderer::Type::Text:
			{
				sf::Text *text = renderer.data.text;
				text->setPosition(pos);
				window.draw(*text);
			}
			break;
		};

	}
};