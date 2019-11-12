#include <stdio.h>
#include <cplx.h>
#include "ComputeFreeEnergy.h"
#include "LocalGreens.cu"


template <typename T>
__device__ void computeFreeEnergyLocal( 
	const cplx<T>& D, const T& e_im, const cplx<T>& gamma, const cplx<T>& gammabar, cplx<T>& free_e )
{
	cplx<T> i((T)0.0,(T)1.0);

	cplx<T> f,fb,g;
	
	Greens<T>::computeGreens_local(					gamma, gammabar, g );
	GreensAnomalous<T>::computeGreens_local(		gamma, gammabar, f );
	GreensAnomalousTilde<T>::computeGreens_local(	gamma, gammabar, fb );

	free_e = M_PI * (D.conj()*f + D*fb) / (T(1.0) + i*g);
}


template <typename T>
__global__ 
void computeFreeEnergy_Kernel(
	const int N, const int Ne,
	T *free_re, T *free_im,
	T *gamma_re, T *gamma_im,
	T *gammabar_re, T *gammabar_im,
	char *domainLabels,
	T *D_re, T *D_im,
	T *e_im, T *residues )
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;		// x index
	const int y = blockDim.y * blockIdx.y + threadIdx.y;		// y index
	
	int gamma_idx, free_idx;
	int e_idx;
	
	if ( x < N )
	{
		free_idx = y*N + x;
	
		cplx<T> free_sum(T(0.0),T(0.0)), freeTmp, gamma, gammabar;

		const cplx<T> D(D_re[free_idx], D_im[free_idx]);
	
		if ( domainLabels[free_idx] > 0 )
		{
			for ( e_idx=0; e_idx<Ne; ++e_idx )
			{
				gamma_idx = e_idx*2*N*N + free_idx;

				gamma.x = gamma_re[gamma_idx];
				gamma.y = gamma_im[gamma_idx];

				gammabar.x = gammabar_re[gamma_idx];
				gammabar.y = gammabar_im[gamma_idx];

				computeFreeEnergyLocal<T>(
					D, e_im[e_idx],
					gamma, gammabar,
					freeTmp );
		
				free_sum += residues[e_idx] * freeTmp;
			}		
		}
	
		free_re[free_idx] += free_sum.x;
		free_im[free_idx] += free_sum.y;
	}
}


template <typename T>
ComputeFreeEnergy<T>::ComputeFreeEnergy()
	: ComputeProperty<T>(), _freeEnergyGrid(0)//, _tmp(0)
{}


template <typename T>
ComputeFreeEnergy<T>::ComputeFreeEnergy( Context<T>* ctx )
	: ComputeProperty<T>(ctx), _freeEnergyGrid(0)//, _tmp(0)
{
	ctx->add(this);
}


template <typename T>
ComputeFreeEnergy<T>::~ComputeFreeEnergy()
{
	if ( _freeEnergyGrid ) delete _freeEnergyGrid;
}


template <typename T>
int ComputeFreeEnergy<T>::initialize()
{
	std::cout << "ComputeFreeEnergy<T>::initialize()\n";
	
	typedef ComputeProperty<T> C;
	
	// Get resolution
	const int N0 = C::getParameters()->getGridResolutionBase();
	const int N1 = C::getParameters()->getGridResolutionCoherence();

	GridGPU::initializeGrid(_freeEnergyGrid, N0, N0);
	
	C::initGreensAngle( N1 );

	return 0;
}


template <typename T>
IntegrationIterator<T>* ComputeFreeEnergy<T>::createIntegrationIterator( int numEnergiesPerBlock )
{
	return new IntegrationIteratorOzaki<T>( ContextModule<T>::getContext(), numEnergiesPerBlock );
}


template <typename T>
void ComputeFreeEnergy<T>::initializeIteration( int numFields )
{
	typedef ContextModule<T> C;
	
	if ( ComputeProperty<T>::doCompute() )
	{
		_freeEnergy_angle = 0.0;
		_freeEnergy_im = 0.0;
		
		_freeEnergyGrid->setZero();
	}
}


template <typename T>
void ComputeFreeEnergy<T>::initializeGreensAngle()
{
	if ( ComputeProperty<T>::doCompute() )
		ComputeProperty<T>::initializeGreensAngle();	// Sets to zero
}


template <typename T>
void ComputeFreeEnergy<T>::computeGreensAngle(
	ParameterIterator<int>* iter, GridGPU* gamma, GridGPU* gammabar, GeometryGroup<T>* geom )
{
	// Computes 'I' for current angle, for every spatial grid point (summed over energies).
	typedef ContextModule<T> C;

	if ( ComputeProperty<T>::doCompute() )
	{
		IntegrationIterator<T>* ii = C::getIntegrationIterator();

		GridGPU* result = ComputeProperty<T>::getGreensCurrentAngle();

		GridGPU* deltaRot	= ii->getOrderParameterFieldCurrentAngle();
		GridGPU* energies	= ii->getEnergies();
		GridGPU* residues	= ii->getResidues();
		
		int numEnergies		= iter->getNumInterval();
		int energyIdx_start	= iter->getCurrentValue();
	
		dim3 bpg, tpb;
		result->getKernelDimensions( bpg, tpb );
		
		computeFreeEnergy_Kernel<T><<<bpg,tpb>>>(
			result->dimX(), numEnergies,
			result->getDataPointer( 0, Type::real ),
			result->getDataPointer( 0, Type::imag ),
			gamma->getDataPointer( 0, Type::real ),
			gamma->getDataPointer( 0, Type::imag ),
			gammabar->getDataPointer( 0, Type::real ),
			gammabar->getDataPointer( 0, Type::imag ),
			geom->getDomainLabelsGrid()->getDataPointer( 0, Type::real ),
			deltaRot->getDataPointer( 0, Type::real ),
			deltaRot->getDataPointer( 0, Type::imag ),
			energies->getDataPointer( 0, Type::imag ) + energyIdx_start,
			residues->getDataPointer( 0, Type::real ) + energyIdx_start
			);
	}
}


template <typename T>
void ComputeFreeEnergy<T>::integrateGreensAngle( GeometryGroup<T>* geom, const T angle )
{
	// Sums 'I' over spatial grid points, and adds result to total angle integration.
	// (Variable _freeEnergy_angle is 'I')
	if ( ComputeProperty<T>::doCompute() )
	{
		GridGPU* greens = ComputeProperty<T>::getGreensCurrentAngle();

		geom->extendValuesAtBoundary( greens, 2 );
		_freeEnergyGrid->insertRotatedGridAdditive( greens, angle );
	}
}


template <typename T>
void ComputeFreeEnergy<T>::computePropertyFromGreens()
{
	typedef ContextModule<T> C;

	if ( ComputeProperty<T>::doCompute() )
	{
		// Get properties
		int numAngles	= C::getParameters()->getAngularResolution();
		T h				= C::getParameters()->getGridElementSize();
		T T_Tc			= C::getParameters()->getTemperature(); // T/Tc
		T B				= C::getParameters()->getMagneticFieldStrength();
		T area			= C::getGeometry()->area();
		T areaD			= C::getGeometry()->area(1);
		

		// Compute second term of free energy not involving angle integration
		GridCPU energies = *(C::getIntegrationIterator()->getEnergies());
		GridCPU residues = *(C::getIntegrationIterator()->getResidues());
		//int numEnergies = energies.getNumGridElements();
		int numEnergies = energies.dimX();
	
		T *e_im = energies.getDataPointer(0,Type::imag);
		T *r_re = residues.getDataPointer(0,Type::real);
		T e_inv_sum = 0.0;

		for ( int i=0; i<numEnergies; ++i )
			e_inv_sum += r_re[i]/e_im[i];
	
		// Compute free energy new way
		T freeNew_re = 0.0, freeNew_im = 0.0;
		C::getGeometry()->zeroValuesOutsideDomain( _freeEnergyGrid );
		_freeEnergyGrid->sumField( freeNew_re, freeNew_im );
		

		// Compute total free energy
		double freeEnergy_unitArea_old = _freeEnergy_unitArea;

		if ( abs(freeEnergy_unitArea_old) < 0.000000001 )
			freeEnergy_unitArea_old = 0.000000001;

		OrderParameter<T>* delta = C::getOrderParameter();
		const GridGPU& delta_field = *(delta->getOrderParameterField());

		_freeEnergy_unitArea = -2.0/(T)numAngles * T_Tc * freeNew_re;

		for ( int i=0; i<delta_field.numFields(); ++i )
		{
			double mult = log(T_Tc / delta->getOrderParameterComponent(i)->getRelativeTc()) + 2.0*M_PI * T_Tc * e_inv_sum;
			_freeEnergy_unitArea += delta_field[i].complexMagnitudeSq().sumField() * mult;
		}

		_freeEnergy = _freeEnergy_unitArea * h*h;
		_freeEnergy_unitArea *= 1.0/areaD;


		// Set residual and convergence
		double residual = (_freeEnergy_unitArea - freeEnergy_unitArea_old)/ freeEnergy_unitArea_old;
		ComputeProperty<T>::setResidual( residual );

		if ( residual < C::getParameters()->getConvergenceCriterion() )
			ComputeProperty<T>::setConverged();

		using namespace std;
	
		cout << "Area = " << C::getGeometry()->area() << endl;
		cout << "_freeEnergy_unitArea = " << _freeEnergy_unitArea << endl;
	}
	else
	{
		_freeEnergy = 0.0;
	}
}


template <typename T>
T ComputeFreeEnergy<T>::freeEnergy_S( Parameters<T>* param,
	GeometryGroup<T>* geom, OrderParameter<T>* s_wave )
{
	GridGPU* delta = s_wave->getOrderParameterField();
	
	// Temperature
	T T_Tc = param->getTemperature();
	
	// Area
	T area = geom->area();
	
	// |Delta|^2
	GridGPU deltaValue = delta->complexMagnitudeSq();
	cplx<T> D2 = geom->fieldAverage( &deltaValue );
	
	// Sum	
	GridCPU* energies = param->getEnergies();	
	int numEnergies = energies->dimX();

	T *e_ptr = energies->getDataPointer(0,Type::imag);
	T e_inv_sum = 0.0, tmp, e_n;

	for ( int i=0; i<numEnergies; ++i )
	{
		e_n = e_ptr[i];
		tmp = e_n + sqrt(D2.x + e_n*e_n);
		e_inv_sum += 1.0 / (e_n * tmp*tmp);
	}
	
	// Free energy
	T freeEnergy = area*D2.x * ( log(T_Tc) + 2.0*PI*T_Tc*D2.x*e_inv_sum );
	return freeEnergy;
}


template <typename T>
T ComputeFreeEnergy<T>::freeEnergyT0_S()
{
	// Temperature
	T T_Tc = 0.01;
	
	// Area
	T area = 1.0;//geom->area();
	
	// Delta at T=0
	T D0 = Initial<T>::computeDeltaBulk( T_Tc ); // = 1.76

	// Free energy
	return area*(-0.5*D0*D0 + PI*PI/3.0 * T_Tc*T_Tc);
}






