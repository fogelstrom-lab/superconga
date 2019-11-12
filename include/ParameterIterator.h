#ifndef _PARAMETERITERATOR_H
#define _PARAMETERITERATOR_H


template <typename T>
class ParameterIterator
{
public:
			ParameterIterator();
			~ParameterIterator() {}
		
			ParameterIterator( T p_min, T p_max );		
	
	ParameterIterator<T>& operator=( const ParameterIterator<T>& right );
	T operator()() { return getCurrentValue(); }
	
	T		getMin() const { return _p_min; }
	T		getMax() const { return _p_max; }
	T		getIncrement() const { return _p_inc; }
	T		getRange() const { return _p_max - _p_min; }
	int		getNum() const { return _p_num; }
	T		getNumInterval() const;
	int		getSweepDirection() const { return _sweepDir; }
	int		getIndex() const { return _count; }
	
	void	setMin( T p_min );
	void	setMax( T p_max );
	void	setIncrement( T p_inc );
	void	setNum( int p_num );
	void	setSweepDirection( int sweepDir );
	void	setNull();	// Do not iterate at all
	void	setSingle();	// Iterate once
	void	setSingleValue();	// Iterate once
	
	void	begin( int idx = 0 );
	T		next( int step = 1 );
	int		isDone() const { return _isDone; }
	
	T		getCurrentValue() const;
	
private:

	void updateIncrement();
	void updateNumber();

	T _p_min, _p_max, _p_inc;
	int _p_num;
	
	int _sweepDir;
	int _count;
	int _isDone;
};

#endif

