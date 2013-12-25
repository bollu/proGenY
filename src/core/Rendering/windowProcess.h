#pragma once
#include "../controlFlow/Process.h"
#include "../controlFlow/processMgr.h"
#include "../include/SFML/Graphics.hpp"


/*!represents the window. 
Is responsible for double buffering*/  
class windowProcess : public Process{
	sf::RenderWindow *window;
	sf::Color clearColor;
public:
	windowProcess(processMgr &processManager, Settings &settings, EventManager &eventManager) : 
	Process("windowProcess"){
		
		vector2 *screenDim = settings.getPrimitive<vector2>(Hash::getHash("screenDimensions"));

		sf::ContextSettings context;
		context.antialiasingLevel = 8;

		const sf::VideoMode &fullscreenMode = sf::VideoMode::getFullscreenModes()[0];
		const sf::VideoMode &desktopMode = sf::VideoMode::getDesktopMode();

		//actually fill this up..
		if(true){
			this->window = new sf::RenderWindow(desktopMode, 
			"ProGenY", sf::Style::Default, context);
		}else{
			this->window = new sf::RenderWindow(fullscreenMode, 
			"ProGenY", sf::Style::Fullscreen, context);
		}
		
		clearColor = sf::Color(0, 0, 0, 255);
	}

	/*!clears the window and gets it ready for rendering */
	virtual void preUpdate(){
		window->clear(this->clearColor);
	};
	
	/*!flips the buffer and renders the back buffer */
	void postDraw(){
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
