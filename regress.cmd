echo "--------- Testing Fuel Efficiency ------------"

sa.engine -f SA -x -O SA/models/fuel_efficiency/dataset.test.osql -o "quit;"
if %ERRORLEVEL% neq 0 goto fail

sa.engine -f SA -x -O SA/models/fuel_efficiency/inference.test.osql -o "quit;"
if %ERRORLEVEL% neq 0 goto fail

sa.engine -f SA -x -O SA/models/fuel_efficiency/weights.test.osql -o "quit;"
if %ERRORLEVEL% neq 0 goto fail

goto ok
:fail
echo "******************************************************"
echo "********* ERROR IN REGRESS *********"
echo "******************************************************"
echo errorlevel is %ERRORLEVEL%
exit /b %ERRORLEVEL%

:ok
exit /b 0
