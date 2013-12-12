#pragma once
#include "Process.h"
#include "processMgr.h"
#include "../include/SFML/Graphics.hpp"


/*!represents the window. 
Is responsible for double buffering*/  
class windowProcess : public Process{
	sf::RenderWindow *window;
	sf::Color clearColor;
public:
	windowProcess(processMgr &processManager, Settings &settings, eventMgr &eventManager) : 
	Process("windowProcess"){
		
		vector2 *screenDim = settings.getProp<vector2>(Hash::getHash("screenDimensions"));

		sf::ContextSettings context;
		//context.antialiasingLevel = 8;
		/*this->window = new sf::RenderWindow(sf::VideoMode(screenDim->x, screenDim->y), 
			"ProGenY", sf::Style::Default, context);
		*/

		this->window = new  sf::RenderWindow(sf::VideoMode(800, 600), "SFML window");
		clearColor = sf::Color(0, 0, 0, 255);
	}

	/*!clears the window and gets it ready for rendering */
	virtual void preUpdate(){
		window->clear(this->clearColor);
	};
	
	/*!flips the buffer and renders the back buffer */
	void postDraw(){
	//	util::msgLog("-------------------------Flippingwindow");
		window->display();
	}

	void Shudown(){
		window->close();
	}
	
	sf::RenderWindow *getWindow(){
		return this->window;
	}

	void setClearColor(sf::Color color){
		this->clearColor = color;
	}




};