test-inline:	inline.cpp
	hipcc -o test-inline inline.cpp

run-inline: test-inline
	./test-inline

.PHONY:	clean

clean:	
	rm test-inline
