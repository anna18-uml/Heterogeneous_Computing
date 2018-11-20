void global_reduction(__local float*,__global float*);
                

__kernel void pi_cal(__global int *id,
     int num_to_work, __local float* local_result, 
     __global float* global_result) 
     {

	int local_id=get_local_id(0);
	float partial_sum=0.0f;
	char sign=0;
	int num_work_items=get_local_size(0);
	for(int i=0;i<num_to_work;i++)
	{
		if(sign==0)
		{
			partial_sum+=(1.0f/((2*(*id))+1.0f));
			sign=1;
		}
		else
		{
			partial_sum+=(-1.0f/((2*(*id))+1.0f));
			sign=0;
		}
		(*id)=(*id)+1;
	}
	local_result[local_id]=partial_sum;
	barrier(CLK_GLOBAL_MEM_FENCE);
	global_reduction(local_result,global_result);
      }
/*Function for global reduction*/
void global_reduction(__local float* local_result, __global float* global_result)
{
	int num_work_items=get_local_size(0);
	int local_id=get_local_id(0);
	float sum;
	if(local_id==0)
	{
		sum=0.0f;
		for(int i=0;i<num_work_items;i++)
		{
			sum+=local_result[i];
		}
		*global_result=sum;
	}

}
