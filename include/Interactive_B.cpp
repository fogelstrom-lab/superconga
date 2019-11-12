int g_y = 0;
T g_B = 0.0;
Parameters<T>* g_param_ptr = 0;

void registerInteractiveParameters( Parameters<T>* param_in )
{
	g_param_ptr = param_in;
}


void interactiveB_mouseFunc( int button, int state, int x, int y )
{
	if ( state == 0 )
	{
		g_y = y;
		if ( g_param_ptr )
		g_B = g_param_ptr->getMagneticFieldStrength();
	}
}


void interactiveB_motionFunc( int x, int y )
{
	T B = g_B + (y-g_y)/1000.0;
	if ( g_param_ptr )
	{
		g_param_ptr->setMagneticFieldStrength(B);
		std::cout << "B = " << B << "\n";
	}
	else
		std::cout << "Parameters object needs to be registered with: registerInteractiveParameters( Parameters<T>* )\n";
}
