#pragma once
#include "renderProcessor.h"

Renderer::Renderer(sf::Shape *shape){
	this->data.shape = shape;
	this->data.shape->setFillColor(sf::Color(0, 0, 0));
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

renderProcessor::renderProcessor(processMgr &processManager, 
	Settings &settings, eventMgr &_eventManager){

	this->window = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"))->getWindow();
	this->view = processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));

	windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
	windowProc->setClearColor(sf::Color(255, 255, 255, 255));

};


void renderProcessor::onObjectAdd(Object *obj){

}

void renderProcessor::Process(float dt){

	for(auto it= objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;

		renderData *data = obj->getProp<renderData>(Hash::getHash("renderData"));


		if(data == NULL){
			continue;
		}


		//for box2d, +ve x axis is 0 degree clockwise
		//ffor SFML, -ve y axis 0 degree. weird...
		//box2dClockwise = 360 - box2d



		static util::Angle game2RenderAngle = util::Angle::Deg(360);

		vector2* gamePos = obj->getProp<vector2>(Hash::getHash("position"));
		vector2 viewPos = view->game2ViewCoord(*gamePos);
		vector2 renderPos = view->view2RenderCoord(viewPos);

		util::Angle *angle = obj->getProp<util::Angle>(Hash::getHash("facing"));
		float fAngle = angle->toDeg();
		util::Angle gameAngle = game2RenderAngle - *angle;
		

		this->_Render(renderPos, gameAngle, data, data->centered);
	};
}


void renderProcessor::_Render(vector2 pos, util::Angle &angle, 
	renderData *data,  bool centered){
		//loop through the renderers
	for(auto it = data->renderers.begin(); it != data->renderers.end(); ++it){
		Renderer &renderer = *it;


		switch(renderer.type){
			case Renderer::Type::Sprite:
			{
				sf::Sprite *sprite = renderer.data.sprite;
				sprite->setPosition(pos);
				sprite->setRotation(angle.toDeg());
				window->draw(*sprite);
			
			}
			break;


			case Renderer::Type::Shape:
			{
				sf::Shape *shape = renderer.data.shape;


				
				if(centered){
					sf::FloatRect AABB = shape->getLocalBounds();
					//pos -= vector2(AABB.width, AABB.height) * 0.5;
				}
						
				shape->setPosition(pos);
				shape->setRotation(angle.toDeg());
				window->draw(*shape);
			}
			break;

			case Renderer::Type::Text:
			{
				sf::Text *text = renderer.data.text;
				text->setPosition(pos);
				//text->setRotation(angle.toDeg());
				window->draw(*text);
			}
			break;

			default:
				util::errorLog("renderer of unkown type detected.");
		};

	}
};