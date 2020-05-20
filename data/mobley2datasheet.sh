#!/bin/bash


awk ' OFS = "," {if (NR <= 7) {print $0} 
                 else if (NR i==8 ) {print "wind","sza","vza","azi","rho"} 
                 else if ( $1 == "rho" ){wind = $6; sza=$10}
	         else {print wind, sza, $3,  $5, $6 }}  ' rhoTable_Mobley1999.txt > rhoTable_Mobley1999.csv 


# in case of bugs, try dos2unix rhoTable_Mobley2015.txt
awk  ' OFS = "," {if (NR <= 8) {print $0} 
                 else if (NR <20 ) {}
		 else if (NR ==20 ) {print "wind","sza","vza","azi","rho"} 
                 else if ( $1 == "WIND" ) {wind = $4; sza=$9}
                 else {print wind, sza, $1, $2,$3 }}  ' rhoTable_Mobley2015.txt > rhoTable_Mobley2015.csv     


