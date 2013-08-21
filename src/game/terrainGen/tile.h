#pragma once
#include "../../core/vector.h"
#include <set>
namespace terrainGen{

	class Tile;

	typedef int tileType; 


	class Tile{
		
	public:
		Tile(vector2 position, tileType type, bool filled){
			this->position = position;
			this->filled = filled;
			this->type = type;
		}

		vector2 getPosition(){
			return this->position;
		}

		tileType getType(){
			return this->type;
		}

		bool isFilled(){
			return this->filled;
		}

		void setFilled(bool filled){
			this->filled = filled;
		}

		void setType(tileType type){
			this->type = type;
		}

		bool operator == (const Tile &other){
			return this->position == other.position && 
			this->type == other.type && 
			this->filled == other.filled; 
		};

		bool operator != (const Tile &other){
			return !(*this == other);
		};


	private:
		vector2 position;
		bool filled;

		tileType type;
	};



};