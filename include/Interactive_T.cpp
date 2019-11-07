// OpenGL functions for interactive control
/*
int buttonPressed;

void setParamValue( int x, int y )
{
	int winY = vis->getNumPixelsWindowY();
	
	double value = ((double)y/(double)winY) * 0.25 * 50;

	if ( buttonPressed == 0 )
	{
		param->setMagneticFieldStrength( value );
		std::cout << "B = " << value << "\n";
	}
}


void myMouseFunc( int button, int state, int x, int y )
{
	buttonPressed = button;
	setParamValue( x, y );
}


void myMotionFunc( int x, int y )
{
	setParamValue( x, y );
	acc->resetCount();
}
*/

int g_y = 0;
T g_T = 0.0;
Parameters<T>* g_param_ptr = 0;

void registerInteractiveParameters( Parameters<T>* param_in )
{
	g_param_ptr = param_in;
}


void interactiveT_mouseFunc( int button, int state, int x, int y )
{
//	T B = (y-g_y)/1000.0;
//	g_y = y;
//	param->setMagneticFieldStrength(B);
//	std::cout << "B = " << B << "\n";
//	std::cout << "state = " << state << "\n";
	if ( state == 0 )
	{
		g_y = y;
		if ( g_param_ptr )
		g_T = g_param_ptr->getTemperature();
	}
}


void interactiveT_motionFunc( int x, int y )
{
	T T = g_T + (y-g_y)/10000.0;
	
	if ( T < 0.1 )
	{
		T = 0.1;
	}

//	g_y = y;
	if ( g_param_ptr )
	{
		g_param_ptr->setTemperature(T);
		std::cout << "T = " << T << "\n";
	}
	else
		std::cout << "Parameters object needs to be registered with: registerInteractiveParameters( Parameters<T>* )\n";
}
