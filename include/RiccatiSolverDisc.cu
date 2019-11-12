#include "RiccatiSolverDisc.h"
#include "RiccatiSolverDisc_Kernel.cu"

template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverDisc<T,GAMMA,POTENTIAL>::RiccatiSolverDisc()
	:	RiccatiSolver<T>()
{}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverDisc<T,GAMMA,POTENTIAL>::~RiccatiSolverDisc()
{
	std::cout << "~RiccatiSolverDisc()\n";
}

template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
RiccatiSolverDisc<T,GAMMA,POTENTIAL>::RiccatiSolverDisc( Context<T>* ctx )
	:	RiccatiSolver<T>( ctx )
{
	std::cout << "RiccatiSolverDisc::RiccatiSolverDisc( Context<T>* )\n";
	ctx->set( this );
}


template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
int RiccatiSolverDisc<T,GAMMA,POTENTIAL>::initialize()
{
	std::cout << "RiccatiSolverDisc::initialize()\n";
	RiccatiSolver<T>::initialize();

	return 0;
}