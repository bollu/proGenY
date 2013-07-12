#pragma once
#include <random>

class Generator{
protected:
	Generator();
	std::mt19937 generator;
public:
	virtual ~Generator();
};