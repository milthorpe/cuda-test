test-inline:	inline.cpp
	hipcc -o test-inline inline.cpp

run-inline: test-inline
	./test-inline

kernel.hip: kernel.hip.cpp
	hipcc --genco -o kernel.hip kernel.hip.cpp

test-outline:	kernel.hip outline.cpp
	hipcc -o test-outline outline.cpp

run-outline: test-outline
	./test-outline ./kernel.hip

.PHONY:	clean

clean:	
	rm test-inline test-outline
