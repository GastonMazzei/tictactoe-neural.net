-datos originales: results.pkl

-processer.py   (producen x.npy y le cambias el nombre a x.npy.old)

-processer_onlywin.py
-processer_onlywin2.py    (ambos sobre x.npy.old)

-loader.py   (model-onlywin)
-loader2.py  (model-onlywin2)

-game_generator.py  (enhaceresults.pkl)



-processer_for_fine.py  (xfine.npy from enhaceresults.pkl)

-loader_for_fine.py    (model-onlywin-fine from xfine.npy)
-loader2_for_fine.py   (model-onlywin2-fine from xfine.npy) 


-python3 mover9.py       (play against model-onlywin)
-python3 mover9-fine.py  (play against model-onlywin-fine)

-game_generator-it.py  (enhaceresults.pkl)

HOW TO FINE-TUNE
https://stackoverflow.com/questions/57356645/how-to-fine-tune-a-loaded-model-in-keras


python3 processer_for_fine.py ; python3 loader_for_fine.py ; python3 loader2_for_fine.py





*******CASOS DE USO************

for f in "processer" "processer_onlyloose" "processer_onlywin" "loader" "loader2"; do
	python3 scripts/$f.py;
done



./generate 0 0 1 2000


for (( i=0; i<5; ++i)); do
	python3 scripts/processer_for_fine.py
	python3 scripts/loader_for_fine.py
	python3 scripts/loader2_for_fine.py
	python3 game_generator.py 0 0 1 4000 1
done

for i in 



python3 play.py model-onlywin



 
python3  processer.py ; python3 processer_onlyloose.py; python3 processer_onlywin.py ; python3 loader.py; python3 loader2.py 
python3 processer_for_fine.py ; python3 loader_for_fine.py ; python3 loader2_for_fine.py ; python3 game_generator-it.py ;



