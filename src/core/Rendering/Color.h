#pragma once


struct Color {
private:
	float r_, g_, b_, a_;

	float normalizeComponent (float component) {
		while(component > 1.0) {
			component -= 1.0;
		}
		while (component < 0.0) {
			component += 1.0;
		}

		return component;
	}
	Color(float r, float g, float b, float a) {
		r_ = normalizeComponent(r);
		g_ = normalizeComponent(g);
		b_ = normalizeComponent(b);
		a_ = normalizeComponent(a);
	};

public:

	static Color Int(uint8_t r, uint8_t g, uint8_t b, uint8_t a){
		return Color(r / 255.0, g / 255.0, b / 255.0, a / 255.0);
	};

	static Color Float(float r, float g, float b, float a) {
		return Color(r, g, b, a);
	};

	uint8_t ir() { return r_ * 255.0f; };
	uint8_t ig() { return g_ * 255.0f; };
	uint8_t ib() { return b_ * 255.0f; };
	uint8_t ia() { return a_ * 255.0f; };
	uint8_t io() { return (1.0f - a_) * 255.0f; };


	float fr() {  return r_; };
	float fg() {  return g_; };
	float fb() {  return b_; };
	float fa() {  return a_; };
	float fo() {  return (1.0f - a_); };

	void toArray(float arr[4]) {
		arr[0] = r_;
		arr[1] = g_;
		arr[2] = b_;
		arr[3] = a_;
	}

	void toArray(int arr[4]) {
		arr[0] = ir();
		arr[1] = ig();
		arr[2] = ib();
		arr[3] = ia();
	}

	Color operator +(const Color &other) const {
		return Color(r_ + other.r_, 
			g_ + other.g_, 
			b_ + other.b_, 
			a_ + other.a_);
	}

	Color operator -(const Color &other) const {
		return Color(r_ - other.r_, 
			g_ - other.g_, 
			b_ - other.b_, 
			a_ - other.a_);
	}

	Color operator *(const float f) const {
		return Color(r_ * f, g_ * f, b_ * f, a_ * f);
	}

	Color operator /(const float f) const {
		return Color(r_ / f, g_ / f, b_ / f, a_ / f);
	}
};

