
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Function to parse mapping data in "Encoded Value\tCategorical Value" format
def parse_mapping(data):
    mapping = {}
    for line in data.split('\n'):
        if line.strip():  # Skip empty lines
            encoded, category = line.split('\t')
            mapping[category.strip()] = int(encoded.strip())
    return mapping

# Example mapping data (replace with your actual mappings)
brand_data = """14	Ford
4	BMW
21	Jaguar
40	Pontiac
0	Acura
3	Audi
15	GMC
31	Maserati
9	Chevrolet
41	Porsche
35	Mercedes-Benz
50	Tesla
26	Lexus
23	Kia
27	Lincoln
11	Dodge
52	Volkswagen
25	Land
8	Cadillac
33	Mazda
42	RAM
48	Subaru
19	Hyundai
30	MINI
22	Jeep
17	Honda
18	Hummer
38	Nissan
51	Toyota
53	Volvo
16	Genesis
37	Mitsubishi
7	Buick
20	INFINITI
34	McLaren
47	Scion
24	Lamborghini
5	Bentley
49	Suzuki
13	Ferrari
1	Alfa
44	Rolls-Royce
10	Chrysler
2	Aston
43	Rivian
28	Lotus
46	Saturn
29	Lucid
36	Mercury
32	Maybach
12	FIAT
39	Plymouth
6	Bugatti
45	Saab
54	smart"""

model_data = """649	F-150 Lariat
49	335 i
1803	XF Luxury
1779	X7 xDrive40i
699	Firebird Base
889	Integra LS
1443	S5 3.0T Prestige
245	Acadia SLT-1
166	A3 2.0T Tech Premium
983	MDX Technology
1247	Quattroporte S Q4 GranLusso
1641	Tahoe Premier
424	Cayenne S
1224	Q7 55 Prestige
331	C-Class C 300 4MATIC Sport
1656	Terrain SLT-1
1063	Model 3 Long Range
909	LS 460 Base
1520	Sierra 2500 SLE
1650	Telluride LX
1654	Terrain SLE
1685	Transit-350 XLT
996	MKZ Base
1094	Mustang EcoBoost Premium
434	Challenger GT
506	Corvette Z06
628	Expedition Max XLT
194	A7 Premium
893	Jetta 1.4T SE
674	F-350 King Ranch
1681	Transit Connect XLT
957	M340 i xDrive
1516	Sierra 1500 SLT
169	A4 2.0T Premium Plus quattro
1399	Rover Range Rover Sport HSE Dynamic
593	Escalade ESV Platinum
1658	Thunderbird Premium
377	CX-30 Premium Package
157	911 R
13	1500 Rebel
1837	Yukon XL SLT
1390	Rover Range Rover P530 SE LWB 7 Seat
173	A4 3.2 Premium Plus quattro
514	Crosstrek 2.0i Limited
443	Challenger SXT
799	Ghibli S Q4 GranSport
660	F-150 XLT SuperCab
1156	Palisade SEL
488	Cooper Base
1643	Taurus SHO
97	528 i xDrive
202	ALPINA B7 Base
1398	Rover Range Rover Sport HSE
1743	Wrangler Unlimited Sport
1736	Wrangler Sport
1369	Rover Range Rover 3.0L Supercharged HSE
670	F-250 XLT
1411	Rover Range Rover Velar P250 SE R-Dynamic
1195	Protege DX
871	IS 350 F Sport
1774	X6 xDrive35i
1478	SQ5 3.0T Premium Plus
543	E-Class E 350 4MATIC
706	Focus SE
804	Gladiator Rubicon
953	M3 Base
727	Fusion SE
1556	Sonata GLS
792	Genesis Coupe 3.8 Base
1730	Wrangler Rubicon
542	E-Class E 350
1298	RX 350 F Sport Performance
1096	Mustang GT Premium
1430	S2000 Base
1771	X6 M Base
659	F-150 XLT
505	Corvette Stingray w/3LT
1082	Model Y Long Range
838	H2 Base
223	AMG GLE 43 Coupe 4MATIC
382	CX-7 Grand Touring
749	GL-Class GL 450 4MATIC
1087	Murano SL
773	GTI Base
1470	SLC 300 Base
392	Camaro 1LT
1378	Rover Range Rover Evoque Pure
167	A4 2.0T Premium
1174	Pathfinder S
1350	Rover Defender V8
1434	S4 3.0 Prestige
1035	Maxima SL
1703	Tundra SR5
290	BRZ Limited
1733	Wrangler SE
1387	Rover Range Rover P525 HSE SWB
1755	X3 xDrive30i
1775	X6 xDrive40i
126	740 iL
1160	Panamera 4 Platinum Edition
1084	Monte Carlo SS
1499	Sequoia Platinum
328	C-Class C 300
1439	S40 T5
627	Expedition Max Limited
745	G80 3.8
664	F-250 Lariat
630	Expedition Timberline
72	435 i
200	A8 L 55
1772	X6 M50i
1162	Panamera Base
932	Land Cruiser Base
283	Avenger SE
1271	RDX Base
518	Cruze LT
1769	X5 xDrive40i
1796	XE 25t
1595	Suburban LT
1472	SLK-Class SLK 350
457	Civic LX
929	Lancer GTS
1423	S-Class S 550 4MATIC
781	GX 470 Base
1640	Tahoe LTZ
1463	SL-Class SL 550
775	GV70 2.5T
1768	X5 xDrive35i
853	Highlander XLE
402	Camry Hybrid XLE
432	Cayman S
969	M6 Base
1682	Transit-250 Base
1573	Sportage LX
981	MDX Sport
793	Genesis Coupe 3.8 Grand Touring
21	228 i
80	4Runner Limited
1433	S3 2.0T Tech Premium Plus
1395	Rover Range Rover Sport 3.0L Supercharged HSE
1431	S2000 CR
1468	SL-Class SL600 Roadster
914	LS V8
840	H3 Base
1013	Magnum Base
1315	Ram 1500 ST
1700	Tundra Limited
1571	Sorento SX Prestige
1329	Renegade Trailhawk
730	G-Class G 550 4MATIC
1436	S4 3.0T Premium Plus
584	Envision Preferred
1742	Wrangler Unlimited Sahara
386	CX-9 Signature
1285	RS 7 4.0T Prestige
346	CLK-Class 550
850	Highlander Limited Platinum
1205	Q5 2.0T Premium Plus
125	740 i
10	1500 Laramie
208	AMG C 63 S
1591	Suburban 1500 LTZ
1319	Ram 2500 SLT Quad Cab
802	Gladiator Mojave
1223	Q7 45 Premium Plus
186	A6 55 Premium
8	1500 Classic Tradesman
395	Camaro 2SS
711	Forester 2.5 XT
398	Camaro Z28
1657	Thunderbird Deluxe
1407	Rover Range Rover Sport Supercharged HSE
141	911 Carrera
393	Camaro 1SS
1568	Sorento Plug-In Hybrid SX Prestige
1664	Titan SV
1466	SL-Class SL500 Roadster
559	ES 350 Base
228	AMG GT 53 Base
1734	Wrangler Sahara
43	328 i xDrive
1464	SL-Class SL 63 AMG
666	F-250 Platinum
737	G37 x
1794	XC90 T6 Momentum
489	Cooper S Base
908	LS 430 Base
165	A3 2.0T Premium
1337	Roma Base
456	Civic EX
1373	Rover Range Rover 5.0L Supercharged Autobiography LWB
1770	X5 xDrive50i
959	M4 Base
577	Enclave Avenir
1012	Macan Turbo
190	A7 3.0T Premium Plus
325	Bronco XLT
138	86 860 Special Edition
1820	XT5 Luxury
1232	Q8 55 Premium Plus
1410	Rover Range Rover Velar P250 S R-Dynamic
187	A6 55 Premium Plus
1058	Metris Base
108	570S Spider
1766	X5 sDrive35i
629	Expedition Platinum
1617	TSX 2.4
4	1500 Big Horn
446	Charger R/T
499	Corvette Base
118	718 Boxster S
1633	Tacoma SR5
1631	Tacoma PreRunner
803	Gladiator Overland
899	Juke SL
1403	Rover Range Rover Sport SVR
943	Liberty Sport
1391	Rover Range Rover P530 SE SWB
733	G35 Sport
474	Compass Sport
600	Escalade Platinum
911	LS 460 L
798	Ghibli S Q4
1858	xB Base
687	F-TYPE R
1015	Malibu 1LT
503	Corvette Stingray w/1LT
816	Golf R 4-Door
657	F-150 Tremor
1532	Silverado 1500 LT Crew Cab
504	Corvette Stingray w/2LT
778	GX 460 Base
751	GL-Class GL 550 4MATIC
272	Atlas 3.6L SEL Premium R-Line
806	Gladiator Sport S
827	Grand Cherokee L Laredo
1383	Rover Range Rover HSE
639	Explorer XLT
1495	Savana 2500 Work Van
562	Eclipse GT
1140	Outback 2.5i Limited
597	Escalade ESV Sport
189	A7 3.0T Premium
1834	Yukon XL 1500 SLT
1255	R8 5.2
275	Avalanche 1500 LT
40	325 i
1388	Rover Range Rover P525 Westminster
1612	TLX Type S PMC Edition
658	F-150 XL
151	911 Carrera S
1644	Taurus X Limited
1197	Q3 2.0T Premium Plus
1101	Mustang Premium
316	Bronco Big Bend Advanced
675	F-350 Lariat
1690	Trax LS
1301	RX 450h Base
547	E-Class E 550
541	E-Class E 300 4MATIC
962	M4 Competition xDrive
17	1500 Tradesman/Express
1757	X4 M Competition
1827	Xterra S
162	A-Class A 220 4MATIC
465	Colorado LT Crew Cab
870	IS 350 Base
1243	Quattroporte Base
1294	RX 350 Base
1740	Wrangler Unlimited Rubicon
634	Explorer Platinum
1412	Rover Range Rover Velar P380 S
1184	Pilot EX-L
538	E-Class AMG E 53
1045	Mazda3 i Touring
342	CC 2.0T Sport
394	Camaro 2LT
239	ATS 2.5L Luxury
38	323 Ci
86	4Runner TRD Pro
131	750 i
1427	S-Class S 63 AMG
653	F-150 Limited
1559	Sonata SE
1075	Model X Base
748	GL-Class GL 450
831	Grand Cherokee Laredo
1011	Macan S
214	AMG E 63 S 4MATIC
1708	Urus Base
1380	Rover Range Rover Evoque S
479	Continental GT V8
895	Jetta GLX VR6
1789	XC90 3.2 Premier Plus
1487	SX4 Base
78	488 Spider Base
1824	XTS Luxury
1342	Romeo Stelvio Quadrifoglio
807	Gladiator Willys
594	Escalade ESV Platinum Edition
1036	Maxima SV
470	Commander Sport
1831	Yukon Denali Ultimate
1268	RC F Base
881	Impreza WRX
999	MKZ Reserve I
358	CT 200h Premium
181	A6 3.0 TDI Premium Plus
174	A5 2.0T Premium
1683	Transit-350 Base
1762	X5 M Base
592	Escalade ESV Luxury
1145	Outback Limited
1364	Rover LR3 HSE
642	Express 1500 Cargo
1311	Ram 1500 Quad Cab
1511	Sierra 1500 Limited Elevation
1136	Optima LX
1283	RS 7 4.0T
747	G90 5.0 Ultimate
1748	X1 xDrive 28i
120	718 Cayman GTS
832	Grand Cherokee Limited
813	Golf GTI 2.0T SE 4-Door
1498	Sequoia Limited
780	GX 460 Premium
1404	Rover Range Rover Sport SVR Carbon Edition
1833	Yukon SLT
965	M5 Base
222	AMG GLE 43 4MATIC Coupe
1157	Panamera 2
1328	Regal Turbo - Premium 1
1154	Palisade Calligraphy
282	Avalon XLS
1537	Silverado 1500 Limited High Country
926	Lancer Evolution GSR
1359	Rover Discovery Sport S R-Dynamic
800	Ghost Base
207	AMG C 63 Base
206	AMG C 43 Base 4MATIC
510	Countryman Cooper S ALL4
784	Gallardo LP560-4
425	Cayenne Turbo
179	A6 2.0T Premium Plus
532	Durango GT
636	Explorer Sport
993	MKX Base
867	IS 250 Base
1422	S-Class S 550
344	CLA-Class CLA 250
175	A5 2.0T Premium Plus
606	Escalade Sport Platinum
18	200 C
198	A8 L 3.0T
365	CTS 2.0L Turbo Luxury
440	Challenger SRT Hellcat
1519	Sierra 2500 Denali
501	Corvette Stingray
329	C-Class C 300 4MATIC
907	LC 500 Base
1454	S7 4.0T Premium Plus
568	Edge Sport
1750	X2 xDrive28i
226	AMG GLE AMG GLE 63 S-Model 4MATIC
56	350Z Enthusiast
846	Highlander Hybrid Limited Platinum
852	Highlander SE
1371	Rover Range Rover 5.0L Supercharged
1541	Silverado 1500 Z71 Extended Cab
251	Accord EX-L
1236	QX60 Base
1814	XK8 Base
954	M3 CS
1652	Telluride SX
353	CR-V EX
1001	MX-5 Miata Base
1693	Tucson SE
1737	Wrangler Unlimited
1692	Tucson Limited
602	Escalade Premium
1842	Z4 3.0i Roadster
902	K5 GT-Line
240	ATS 3.6L Luxury
533	Durango R/T
421	Cayenne GTS
58	350Z Touring
313	Bronco Badlands Advanced
1419	S-10 LS Crew Cab
1720	Viper GTC
1844	Z4 sDrive28i
29	2500 Tradesman
858	Huracan LP610-4
66	428 i xDrive SULEV
942	Liberty Renegade
694	FJ Cruiser Base
869	IS 300 Base
599	Escalade Luxury
119	718 Cayman GT4
76	440 i xDrive
1204	Q5 2.0T Premium
1846	ZDX Base
1122	Navigator Reserve
1340	Romeo Giulia Ti
992	MKT Base
946	M-Class ML 350 4MATIC
1621	TT 1.8T
767	GR86 Premium
923	LaCrosse Leather
264	Arnage R
723	Fusion Hybrid Base
1713	Veloster Turbo R-Spec
1153	Pacifica Touring
662	F-250 King Ranch
1095	Mustang GT
1258	R8 5.2 V10 plus
820	Golf SportWagen TSI SE
1382	Rover Range Rover Evoque SE Premium
324	Bronco Wildtrak Advanced
1704	Tundra SR5 Access Cab
1545	Silverado 2500 LT
1558	Sonata Hybrid Limited
149	911 Carrera Cabriolet
1451	S60 T5 Premier Plus
1623	TT 2.0T Premium
759	GLE 350 Base 4MATIC
1326	Ranger XLT
42	328 i
1826	Xterra Pro-4X
714	Forester Premium
545	E-Class E 400 4MATIC
1221	Q7 3.0T Premium Plus
654	F-150 Platinum
407	Camry Solara SLE V6
88	4Runner Trail
172	A4 2.0T Titanium Premium
400	Camry Hybrid Base
526	Dakota Big Horn/Lone Star
1578	Sprinter 2500
1324	Ranger Lariat
587	Equinox LT
295	Beetle 2.0T S
1588	Stinger GT2
47	330 i xDrive
556	E350 Super Duty XLT
95	525 i
216	AMG G 63 Base
1727	Wagoneer Series III
1597	Suburban Premier
583	Envision Essence
270	Atlas 3.6L SE w/Technology
148	911 Carrera C4S
156	911 GT3 RS
655	F-150 Raptor
417	Cayenne Base
213	AMG E 53 Base 4MATIC
300	Bentayga Speed
1385	Rover Range Rover P400 SE LWB 7 Seat
1560	Sonata Sport 2.0T
1227	Q70 3.7X
1384	Rover Range Rover HSE SWB
1452	S60 T6 Momentum
112	650 Gran Coupe i
1257	R8 5.2 V10 performance
637	Explorer Sport Trac XLT
817	Golf R 4-Door w/DCC & Navigation
872	IS-F Base
1776	X6 xDrive50i
546	E-Class E 450 4MATIC
1566	Sorento LX V6
1370	Rover Range Rover 3.0L V6 Supercharged HSE
306	Boxster Base
145	911 Carrera 4S
132	750 i xDrive
1811	XJ8 L
818	Golf R Base
1051	Mazda6 Signature
865	ILX Technology Plus Package
963	M440 i
1632	Tacoma SR
564	Eclipse Spyder GT
650	F-150 Lariat SuperCrew
1857	tC Release Series 6.0
1104	Mustang V6
341	C70 T5
1120	Navigator L Select
771	GT-R Premium
537	E-Class 400E
1639	Tahoe LT
913	LS 500 F Sport
1179	Phantom
1167	Panamera Turbo
591	Escalade ESV Base
46	330 i
24	2500 Big Horn
1599	Suburban Z71
1098	Mustang Mach-E GT
370	CTS Luxury
1131	Odyssey Elite
1513	Sierra 1500 SLE
1479	SQ5 3.0T Prestige
1749	X1 xDrive28i
843	Hardtop Cooper S
160	911 Turbo Cabriolet
1538	Silverado 1500 Limited LT Trail Boss
1638	Tahoe LS
435	Challenger R/T
50	335 i xDrive
1572	Sorento SXL
114	650 i
260	Armada LE
448	Charger R/T Scat Pack
876	Impreza 2.0i
768	GS 350 Base
1491	Santa Fe SEL
569	Edge Titanium
917	LX 570 Three-Row
574	Element EX
274	Avalanche 1500 LS
1090	Mustang Base
616	Excursion Limited
123	720S Performance
791	Genesis Coupe 2.0T R-Spec
1724	WRX Limited
948	M2 CS
137	850 Turbo
1535	Silverado 1500 LTZ
155	911 GT3
1435	S4 3.0T Premium
319	Bronco Outer Banks Advanced
63	428 Gran Coupe i xDrive
191	A7 3.0T Prestige
1178	Patriot Latitude
725	Fusion Hybrid SE Hybrid
1408	Rover Range Rover Supercharged
1191	ProMaster 2500 High Roof
147	911 Carrera C2S
555	E350 Super Duty Base
1548	Silverado 2500 LTZ H/D Extended Cab
951	M240 i
950	M235 i
183	A6 3.0T Prestige
1505	Sienna LE
301	Bentayga V8
1222	Q7 3.0T Prestige
1793	XC90 T6 Inscription
790	Genesis Coupe 2.0T
1331	Ridgeline RTL-E
215	AMG G 63 4MATIC
385	CX-9 Grand Touring
1413	Rover Range Rover Velar P380 SE R-Dynamic
721	Frontier SV
1705	Tundra SR5 Double Cab
713	Forester Base
1112	NX 300 Base
1754	X3 xDrive28i
1684	Transit-350 XL
71	435 Gran Coupe i
333	C-Class C 63 AMG
1763	X5 M50i
467	Colorado Z85
1581	Sprinter 3500 High Roof
612	Eurovan MV
1425	S-Class S 560 4MATIC
1673	Town Car Base
1593	Suburban 2500 LS
924	Lancer DE
980	MDX 3.7L Advance
1161	Panamera 4S
277	Avalanche LTZ
1060	Mirai Base
1598	Suburban RST
102	535 i xDrive
1741	Wrangler Unlimited Rubicon 392
1240	QX80 Base
626	Expedition Max King Ranch
1097	Mustang Mach-E California Route 1
1185	Pilot Elite
1525	Silverado 1500 2LT
1293	RX 330 Base
987	MKC Base
967	M550 i xDrive
1785	XC60 T6 Inscription
563	Eclipse Spyder GS
35	300M Base
371	CTS Performance
1009	Macan Base
1587	Stinger GT1
671	F-250 XLT Crew Cab Super Duty
30	300 Base
1825	XTS Premium
1181	Pickup Truck XE
1133	Optima EX
1083	Model Y Performance
396	Camaro Base
369	CTS Base
1118	Navigator L
261	Armada Platinum
1164	Panamera GTS
14	1500 SLT
848	Highlander LE Plus
683	F-PACE 35t Premium
1484	SRX Standard
1534	Silverado 1500 LT Trail Boss
423	Cayenne Platinum Edition
1284	RS 7 4.0T Performance Prestige
1608	TLX A-Spec
762	GLK-Class GLK 350
257	Altima 2.5 S
920	LYRIQ Luxury
441	Challenger SRT8
739	G70 2.0T
1052	Mazda6 Touring
144	911 Carrera 4 GTS
204	AMG C 43 AMG C 43 4MATIC
355	CR-V LX
1372	Rover Range Rover 5.0L Supercharged Autobiography
847	Highlander Hybrid XLE
922	LaCrosse CX
877	Impreza 2.0i Premium
258	Altima 2.5 SL
391	California T
1819	XT5 Base
182	A6 3.0T Premium Plus
1504	Shelby GT500 Base
795	Genesis Coupe 3.8 Track
1804	XF Premium
65	428 i xDrive
33	300C Base
413	Caprice Classic Base
828	Grand Cherokee L Limited
715	Forte GT-Line
1214	Q50 3.0t Signature Edition
82	4Runner SR5
136	840 i xDrive
1330	Ridgeline Black Edition
1105	Mustang V6 Premium
1798	XF 25t Premium
406	Camry Solara SLE
408	Camry XLE
1188	Prius Touring
1699	Tundra Hybrid TRD Pro
607	Escape Limited
1756	X3 xDrive35i
318	Bronco Outer Banks
716	Forte Koup EX
823	GranTurismo Sport
161	911 Turbo S
1024	Martin DBX Base
1628	TTS 2.0T Premium Plus
1799	XF 25t Prestige
1836	Yukon XL Denali
1306	RX-8 R3
170	A4 2.0T Premium quattro
753	GLA-Class GLA 250 4MATIC
763	GLS 450 Base 4MATIC
635	Explorer ST
327	C-Class C 250 Sport
476	Continental GT Base
1405	Rover Range Rover Sport Supercharged
62	370Z Touring
279	Avalon Limited
994	MKX Black Label
1163	Panamera Edition
1441	S5 3.0T Premium
1218	Q60 3.0T Premium
1394	Rover Range Rover Sport 3.0 Supercharged HST
466	Colorado Z71
1032	Maverick XLT
20	228 Gran Coupe i xDrive
59	370Z Base
1261	RAV4 Base
809	Golf Auto TSI S
1265	RAV4 TRD Off Road
428	Cayman Base
96	528 i
1812	XK Base
1025	Martin V8 Vantage Base
1745	Wrangler Unlimited X
2	135 i
1709	Utility Police Interceptor Base
732	G35 Base
879	Impreza Outback Sport Wagon
617	Excursion Limited 4WD
195	A7 Premium Plus
611	Escape XLT
522	Cullinan
1183	Pilot EX
83	4Runner SR5 Premium
1606	TL Type S
1386	Rover Range Rover P400 SE SWB
1166	Panamera S
896	Jetta S
150	911 Carrera GTS
604	Escalade Premium Luxury Platinum
1055	Mazda6 iSport VE
585	Eos 2.0T
1802	XF 35t R-Sport
633	Explorer Limited
1618	TSX Base
350	CLS-Class CLS 550
1636	Tahoe Base
1522	Sierra 3500 Denali
1555	Solstice GXP
262	Armada SL
326	C-Class C 250 Luxury
397	Camaro LT1
347	CLK-Class CLK 350
756	GLC 300 Base 4MATIC
334	C-Class C280 4MATIC
1580	Sprinter 2500 Standard Roof
1047	Mazda3 s Sport
416	Cayenne AWD
197	A8 4.0T
111	640 i
1800	XF 3.0 Portfolio
55	3500 Tradesman
1194	ProMaster 3500 Tradesman
779	GX 460 Luxury
1175	Pathfinder SL
1018	Malibu Limited LT
1241	QX80 Luxe
445	Charger GT
770	GT-R Black Edition
530	Dart SE
624	Expedition King Ranch
700	Firebird Trans Am
1438	S4 Base
247	Accent GL
311	Bronco
1565	Sorento LX
586	Equinox 2LT
1614	TLX V6 Advance
60	370Z NISMO
1309	Ram 1500 Laramie
1187	Prius Plug-in Base
645	Express 3500 Base
1666	Titan XD SV
9	1500 Classic Warlock
998	MKZ Reserve
32	300 Touring
561	Eclipse GS
644	Express 2500 Work Van
1533	Silverado 1500 LT Extended Cab
738	G6 GTP
919	LX 600 Premium
1764	X5 PHEV xDrive45e
1723	WRX Base
1426	S-Class S 580 4MATIC
1751	X3 M AWD
1089	Murcielago Base
824	Grand Caravan R/T
315	Bronco Big Bend
551	E-PACE 300 Sport
1615	TLX V6 Tech
1406	Rover Range Rover Sport Supercharged Dynamic
754	GLA-Class GLA 45 AMG
1801	XF 35t Prestige
873	Impala 1LT
787	Genesis 3.8
322	Bronco Sport Big Bend
1276	RL Technology
1345	Routan SE
1344	Romeo Stelvio Ti Sport
1688	Traverse Premier
1447	S6 4.0T Prestige
1377	Rover Range Rover Evoque HSE Dynamic
1219	Q60 3.0t Red Sport 400
1357	Rover Discovery SE
681	F-PACE 25t Premium
180	A6 2.0T Sport
246	Acadia SLT-2
1280	RS 4 Base
1054	Mazda6 i Touring
308	Boxster GTS
233	AMG GT C
192	A7 55 Premium
1489	Santa Fe GLS
1158	Panamera 4
1805	XF S
1605	TL Technology
710	Forester 2.5 X
1622	TT 2.0T
373	CTS-V Base
1680	Transit Connect XL w/Rear Symmetrical Doors
310	Boxster S
625	Expedition Limited
1251	R1S Adventure Package
836	Grand Wagoneer Series III
1752	X3 M40i
25	2500 Laramie
1267	RC 350 F Sport
1467	SL-Class SL550 Roadster
1239	QX70 Base
695	FR-S Monogram
109	640 Gran Coupe i
1786	XC70 3.2
1461	SC 300 Base
1518	Sierra 2500 Base
579	Enclave Leather
1349	Rover Defender SE
988	MKC Reserve
1338	Romeo Giulia Base
1288	RS Q8 4.0T quattro
368	CTS 3.6L Premium Luxury
529	Dakota Sport
1747	X1 sDrive28i
1500	Sequoia SR5
28	2500 SLT
217	AMG G AMG G 63 4MATIC
1510	Sierra 1500 Denali Ultimate
229	AMG GT 63 S 4-Door
544	E-Class E 400
1366	Rover LR4 HSE LUX Landmark Edition
776	GV70 3.5T Sport
689	F-TYPE S British Design Edition
603	Escalade Premium Luxury
81	4Runner Limited Nightshade
490	Cooper S Clubman Base
1334	Rogue SL
769	GS 350 F Sport
1266	RC 350 Base
1502	Shelby GT350 Base
854	Huracan EVO Base
1027	Martin Vantage GT Base
1675	Trailblazer LS
614	Evora 400 Base
1810	XJ8 Base
1141	Outback 2.5i Premium
338	C-Max Energi SE
238	ATS 2.5L
975	M850 i xDrive
1596	Suburban LTZ
52	340 i
1445	S6 4.0T
73	435 i xDrive
482	Continental GT W12
1414	Rover Range Rover Velar R-Dynamic S
379	CX-5 Grand Touring
1531	Silverado 1500 LT
314	Bronco Base
1773	X6 sDrive35i
1432	S3 2.0T Premium Plus
892	Jetta 1.4T S
610	Escape SEL
51	335 is
1738	Wrangler Unlimited 4xe Sahara
632	Explorer Eddie Bauer
1037	Maybach S 580 4MATIC
1813	XK R
955	M3 Competition
1726	WRX STI Base
1557	Sonata Hybrid Base
299	Bentayga S
766	GR86 Base
1792	XC90 Recharge Plug-In Hybrid T8 Inscription 7 Passenger
411	Canyon Elevation Standard
1290	RSX Type S
1855	tC Anniversary Edition
849	Highlander Limited
1123	Navigator Select
104	540 i xDrive
15	1500 Sport
1234	QX56 Base
387	CX-9 Touring
403	Camry LE
536	Durango SRT 392
98	530 i
281	Avalon XLE
500	Corvette Grand Sport
1590	Suburban 1500 LT
1282	RS 5 4.2
834	Grand Cherokee WK Laredo X
1148	Outlander SEL
678	F-350 Platinum
1215	Q50 Hybrid Premium
1111	NX 200t Base
885	Impreza WRX Sti
1281	RS 5 2.9T
1721	Viper SRT-10
367	CTS 3.6L Premium
1574	Sportage Nightfall
1455	S8 4.0T Plus
1678	Trailblazer SS
507	Corvette Z06 w/2LZ
497	Corsair Reserve
712	Forester 2.5i Limited
232	AMG GT Base
573	Elantra SE
302	Bentayga W12 Signature
248	Accent GLS
1165	Panamera Platinum Edition
332	C-Class C 300 Luxury
1603	TL 3.2
676	F-350 Lariat Crew Cab Super Duty DRW
1492	Santa Fe SEL Plus 2.4
422	Cayenne GTS Coupe AWD
292	Baja Base
1130	Odyssey EX-L
427	Cayenne Turbo S
241	Acadia Denali
84	4Runner Sport
1291	RX 300 4WD
783	Gallardo LP550-2
1746	Wrangler X
296	Bentayga Activity Edition
878	Impreza 2.0i Sport
1238	QX60 Pure
1808	XJ Vanden Plas
598	Escalade EXT Base
430	Cayman GTS
580	Enclave Premium
227	AMG GLS 63 4MATIC
1325	Ranger Sport SuperCab
323	Bronco Sport Outer Banks
718	Forte LXS
1317	Ram 2500 Quad Cab
1672	Town & Country Touring-L
708	Focus ST Base
1832	Yukon SLE
1610	TLX PMC Edition
1170	Passat 2.5 SE
1173	Pathfinder Platinum
1102	Mustang SVT Cobra
936	Legacy 2.5 i Premium
698	Fiesta ST
1119	Navigator L Reserve
1564	Sorento Hybrid EX
1147	Outback Touring XT
1259	R8 5.2 quattro Spyder
512	Crossfire Limited
419	Cayenne E-Hybrid S
1273	RDX Technology Package
1440	S5 3.0 Premium Plus
404	Camry SE
1192	ProMaster 2500 Window Van High Roof
1042	Mazda3 Touring
1601	Supra A91 Edition
731	G-Class G 63 AMG
875	Impala Base
1237	QX60 Luxe
669	F-250 XL SuperCab H/D
970	M6 Gran Coupe Base
230	AMG GT AMG GT
549	E-Class E500
1648	Telluride EX
269	Ascent Touring 7-Passenger
1358	Rover Discovery Sport HSE
596	Escalade ESV Premium Luxury Platinum
1362	Rover LR2 Base
483	Continental GTC Base
672	F-250 XLT Super Duty
921	LaCrosse Base
1526	Silverado 1500 Base
493	Corolla LE
127	740e xDrive iPerformance
1400	Rover Range Rover Sport HST MHEV
685	F-PACE SVR
176	A5 2.0T Prestige
1458	S80 3.2
360	CT5 Premium Luxury
122	718 Spyder Base
1124	New Beetle GLS
1019	Malibu Premier
1343	Romeo Stelvio Ti
297	Bentayga Azure First Edition
519	Cruze LT Automatic
647	F-150 FX4
113	650 Gran Coupe i xDrive
1674	Town Car Signature
974	M8 Gran Coupe Competition
436	Challenger R/T Scat Pack
705	Focus RS Base
218	AMG GL AMG GL 63 4MATIC
1841	Z4 2.5i Roadster
1176	Pathfinder SV
883	Impreza WRX Premium
1781	XC40 T5 Momentum
1586	Stinger GT
1220	Q7 3.0T Premium
680	F-350 XLT
1230	Q8 3.0T Prestige
1651	Telluride S
1297	RX 350 F Sport
1655	Terrain SLT
554	E250 Cargo
1482	SRX Luxury Collection
1354	Rover Discovery HSE LUXURY
376	CX-30 Preferred
405	Camry Solara SE
1514	Sierra 1500 SLE Crew Cab
581	Encore GX Essence
1029	Matrix XRS
1563	Sorento EX V6
808	Golf Alltrack TSI SE
1449	S60 R
797	Ghibli S GranLusso
1108	NV Passenger NV3500 HD SV V8
253	Accord Hybrid Base
1446	S6 4.0T Premium Plus
374	CX-30 2.5 S Select Package
1671	Town & Country Touring
928	Lancer Evolution MR
1626	TT RS Base
557	ES 300h Base
1809	XJ6 Vanden Plas
1821	XT5 Premium Luxury
864	ILX Premium Package
517	Cruze LS
399	Camaro ZL1
1250	R-Class R 350 4MATIC
1078	Model X P100D
1552	Sky Base
1453	S7 2.9T Prestige
1396	Rover Range Rover Sport 5.0L Supercharged Dynamic
1333	Roadmaster Estate
1539	Silverado 1500 RST
984	MDX Touring
1530	Silverado 1500 LS
409	Camry XSE
1393	Rover Range Rover SWB
129	750 Li
1016	Malibu LT
855	Huracan EVO Coupe
1584	Sprinter High Roof
278	Avalon Hybrid XLE Premium
1576	Sportage SX Turbo
567	Edge SEL
1462	SC 430 Base
622	Expedition EL Limited
886	Impreza WRX Sti Special Edition
788	Genesis 4.6
1523	Silverado 1500 1LT
492	Corolla Hybrid LE
1830	Yukon Denali
53	3500 Laramie
1125	New Compass Trailhawk
961	M4 Competition
1327	Ranger XLT SuperCab
438	Challenger SRT 392
389	CX-90 Premium
528	Dakota SLT Quad Cab
45	330 330i xDrive
595	Escalade ESV Premium Luxury
1134	Optima Hybrid EX
1007	MX-5 Miata Sport
1609	TLX Base
1553	Sky Red Line
884	Impreza WRX STI
1233	QX30 Premium
1235	QX60 AUTOGRAPH
900	Juke SV
1856	tC Base
947	M2 Base
336	C-Class C55 AMG Sport
937	Legacy 2.5i Premium
851	Highlander Platinum
418	Cayenne Diesel
6	1500 Cheyenne Extended Cab
31	300 S
772	GTC4Lusso T
268	Ascent Limited 7-Passenger
757	GLC 300 GLC 300
1249	Quest SL
734	G37 Journey
1213	Q50 3.0t Red Sport 400
242	Acadia SLE
821	GranTurismo Base
1228	Q70h Base
1172	Passport TrailSport
1689	Traverse RS
531	DeVille Base
193	A7 55 Premium Plus
1469	SL-Class SL63 AMG Roadster
1442	S5 3.0T Premium Plus
941	Liberty Limited
1226	Q70 3.7
1287	RS Q8 4.0T
1277	RLX Advance Package
1085	Monte Carlo Supercharged SS
1049	Mazda6 Grand Touring
525	DTS Luxury II
289	Aviator Reserve AWD
1494	Santa Fe Sport 2.0L Turbo Ultimate
211	AMG CLA 45 Base 4MATIC
812	Golf GTI 2.0T S 4-Door
287	Aventador SVJ Base
468	Colorado ZR2
1217	Q50 Premium
819	Golf SportWagen TSI S 4-Door
99	530 i xDrive
444	Charger Base
199	A8 L 4.0T
1765	X5 eDrive xDrive40e
452	Charger SRT8
266	Arteon 2.0T SEL Premium R-Line
462	Clarity Plug-In Hybrid Base
64	428 i
774	GTO Base
918	LX 600 F SPORT
1759	X4 xDrive30i
1110	NV200 SV
1701	Tundra Platinum
609	Escape SE
973	M8 Competition
472	Compass Latitude
1420	S-Class S 450
777	GV80 2.5T
210	AMG C AMG C 63 S
70	430 i xDrive
540	E-Class E 300
44	328 xi
894	Jetta GLI
473	Compass Limited
185	A6 45 Premium Plus
1542	Silverado 1500 ZR2
1668	Touareg V6 Executive
130	750 Li xDrive
231	AMG GT AMG GT S
159	911 Turbo
956	M3 Competition xDrive
1847	allroad 2.0T Prestige
276	Avalanche 1500 LTZ
502	Corvette Stingray Z51
1753	X3 sDrive30i
48	330e iPerformance
381	CX-50 2.5 Turbo Premium Package
293	Beetle 1.8T
133	750 iL
37	320 i xDrive
1169	Passat 2.0T SE
1031	Maverick XL
454	Cherokee Sport
1314	Ram 1500 SRT-10 Quad Cab
697	FX50 Base
539	E-Class D 2.5 Turbo
1512	Sierra 1500 SL Crew Cab
582	Encore Preferred
571	Elantra HEV Limited
621	Expedition EL King Ranch
841	HS 250h Premium
1310	Ram 1500 Laramie Mega Cab
1066	Model 3 Standard Range Plus
69	430 i
235	AMG S 63 Base 4MATIC
1320	Ram 2500 ST
475	Compass Trailhawk
1787	XC70 T6 Platinum
719	Frontier SE Crew Cab
312	Bronco Badlands
566	Edge SE
1409	Rover Range Rover Supercharged LWB
835	Grand Wagoneer Base
1109	NV200 S
1264	RAV4 SE
110	640 Gran Coupe i xDrive
1788	XC90 3.2
1190	Prius v Three
643	Express 1500 Work Van
1348	Rover Defender S
1517	Sierra 2500 AT4
107	570S Base
1547	Silverado 2500 LTZ
631	Expedition XLT
1503	Shelby GT350R Base
521	Cube 1.8 S
1023	Martin DBS Superleggera
256	Air Grand Touring
164	A3 2.0T
1421	S-Class S 450 4MATIC
345	CLK-Class 500 Cabriolet
3	135 is
511	Coupe Cambiocorsa
1679	Transit Connect XL
1444	S5 4.2 Premium Plus
1245	Quattroporte S
520	Cruze LTZ
703	Flex SEL
1144	Outback 3.6R Touring
1135	Optima Hybrid LX
184	A6 3.2 quattro
337	C-HR LE
447	Charger R/T 392
1312	Ram 1500 SLT Mega Cab
1702	Tundra SR
460	Civic Sport
1043	Mazda3 i SV
995	MKX Reserve
415	Cascada Base
910	LS 460 Crafted Line
565	EcoSport SES
743	G80 2.5T
679	F-350 XL
1460	S90 T5 Momentum
1718	Versa 1.8 S
1843	Z4 3.0si
1028	Matrix XR
1486	STS V6
1263	RAV4 Prime XSE
1339	Romeo Giulia Quadrifoglio
1483	SRX Performance Collection
789	Genesis 5
1026	Martin Vantage Base
1347	Rover Defender 110 SE
1667	Touareg TDI Lux
1300	RX 350 RX 350 F SPORT Handling
916	LX 570 Base
453	Charger Scat Pack
1473	SLK-Class SLK230 Kompressor
1710	V60 Cross Country T5
124	740 Li
1480	SQ7 4.0T
1613	TLX Type S w/Performance Tire
142	911 Carrera 4
1302	RX 450h F SPORT Handling
1242	QX80 SENSORY
1229	Q8 3.0T Premium
1782	XC60 3.2
553	E150 XLT
576	Enclave 1XL
620	Excursion XLT 4WD
1143	Outback 3.6R Limited
949	M2 Competition
548	E-Class E 550 4MATIC
79	4Runner 4WD
11	1500 Limited
785	Gallardo LP570-4 Superleggera
1570	Sorento SX
1744	Wrangler Unlimited Sport S
746	G90 3.3T Premium
1795	XE 20d Prestige
925	Lancer Evolution Base
897	Jetta SportWagen SE
158	911 Targa 4 GTS
1602	Supra A91-MT Edition
375	CX-30 Base
741	G70 3.3T Advanced
451	Charger SRT Hellcat Widebody
1117	Navigator Base
1193	ProMaster 3500 High Roof
811	Golf GTI 2.0T Autobahn 4-Door
623	Expedition EL XLT
833	Grand Cherokee Summit
1067	Model S 100D
339	C30 T5 Premier Plus
1501	Sequoia TRD Pro
842	Hardtop Cooper
140	9-3 Aero
106	550 i xDrive
1363	Rover LR2 HSE
378	CX-30 Select
1475	SLK-Class SLK320
1292	RX 300 Base
1477	SQ5 3.0T Premium
1322	Ram Van 1500
1550	Silverado 3500 High Country
1600	Supra 3.0 Premium
839	H2 SUT
309	Boxster RS 60 Spyder
1367	Rover LR4 Lux
1544	Silverado 2500 High Country
815	Golf R 20th Anniversary Edition
1562	Sorento EX
652	F-150 Lightning XLT
887	Insight EX
615	Evora Base
478	Continental GT Speed
1417	Rover Range Rover Westminster SWB
1121	Navigator Premiere
1077	Model X Long Range Plus
702	Flex Limited
1697	Tundra Hybrid Capstone
1401	Rover Range Rover Sport P400 SE Dynamic
1002	MX-5 Miata Club
1103	Mustang Shelby GT500
801	Gladiator Freedom
1490	Santa Fe SE
1784	XC60 T5 R-Design
964	M440 i xDrive
1335	Rogue SV
351	CLS-Class CLS 63 AMG S-Model 4MATIC
1008	Macan
1211	Q5 S line Premium Plus
1604	TL 3.7
234	AMG GT R
203	ALPINA B7 xDrive
168	A4 2.0T Premium Plus
527	Dakota SLT
1030	Maverick Lariat
1829	Yaris L
1594	Suburban High Country
1729	Wrangler 80th Anniversary
5	1500 Cheyenne
244	Acadia SLE-2
750	GL-Class GL 550
23	240SX Base
1725	WRX Premium
1485	SSR Base
1057	MazdaSpeed3 Grand Touring
487	Convertible John Cooper Works
1044	Mazda3 i Sport
1116	Nautilus Reserve
1687	Traverse LS
249	Accent SEL
143	911 Carrera 4 Cabriolet
410	Canyon Denali
291	BRZ Premium
1686	Traverse High Country
1374	Rover Range Rover 5.0L V8 Supercharged
280	Avalon Touring
12	1500 Longhorn
250	Accord Crosstour EX-L
401	Camry Hybrid LE
1649	Telluride EX X-Line
931	Land Cruiser
1854	i8 Base
1731	Wrangler Rubicon Hard Rock
515	Crosstrek 2.0i Premium
673	F-250 XLT SuperCab Super Duty
39	325 Ci
1004	MX-5 Miata RF Club
1274	RDX w/A-Spec Package
825	Grand Caravan SE
524	DTS Luxury
1663	Titan SL
1279	RS 4 4.2 quattro L
1807	XJ Base
237	ATS 2.0L Turbo Luxury
862	ILX 2.4L
117	718 Boxster GTS
677	F-350 Lariat Super Duty Crew Cab
390	Caliber Express
1760	X5 3.0i
861	ILX 2.0L w/Premium Package
796	Ghibli Base
1767	X5 xDrive 35i Sport Activity
1106	NSX Base
243	Acadia SLE-1
692	F430 Berlinetta
1402	Rover Range Rover Sport SE MHEV
1021	Mark LT Base
1712	Veloster Base
844	Highlander Base
1806	XF XFR-S
1625	TT RS 2.5T
352	CLS-Class CLS500
1000	MKZ Select
372	CTS Premium
589	Equus Signature
1053	Mazda6 i Grand Touring
22	230 i
1524	Silverado 1500 1LZ
469	Commander Base
736	G37 Sport
153	911 Carrera Turbo
461	Civic Type R Touring
139	86 Base
1630	Tacoma Double Cab
952	M240 i xDrive
550	E-Class E55 AMG
1716	Verano Convenience
431	Cayman R
1761	X5 3.0si
1828	Yaris Base
898	Juke NISMO RS
303	Blazer 1LT
945	M-Class ML 350
971	M760 i xDrive
1207	Q5 3.0T Premium Plus
882	Impreza WRX Base
1086	Montero Limited
667	F-250 XL
1392	Rover Range Rover SV Autobiography Dynamic SWB
1459	S80 T6
1020	Marauder Base
704	Flying Spur V8
742	G8 GT
868	IS 250C Base
437	Challenger SE
1739	Wrangler Unlimited Freedom Edition
491	Corolla CE
688	F-TYPE S
36	300ZX Base
1637	Tahoe High Country
1139	Outback 2.5 i Special Edition
1840	Z4 2.5i
1248	Quattroporte Sport GT
1305	RX-8 Grand Touring
1177	Pathfinder Silver
810	Golf GTI 2.0T Autobahn
1474	SLK-Class SLK280 Roadster
1508	Sierra 1500 AT4
1707	Type 57 Base
433	Celica GT
1620	TT 1.8L
1088	Murano SV
720	Frontier SL
1003	MX-5 Miata Grand Touring
115	650 i xDrive
572	Elantra N Base
1046	Mazda3 s Grand Touring
1039	Mazda3 FWD w/Preferred Package
1471	SLK-Class SLK 250
976	MC20 Base
1665	Titan XD S
57	350Z NISMO
450	Charger SRT Hellcat
986	MDX w/Technology Package
1424	S-Class S 560
1022	Martin DB7 Vantage Volante
1048	Mazda6 Carbon Edition
1316	Ram 2500 Laramie Quad Cab
735	G37 S Sport
1159	Panamera 4 Edition
1295	RX 350 Crafted Line F Sport
146	911 Carrera 4S Cabriolet
513	Crosstour EX-L
1216	Q50 Hybrid Sport
814	Golf GTI 2.0T SE w/Performance Package 4-Door
701	Fit Sport
188	A6 55 Prestige
656	F-150 SVT Raptor
1428	S-Class S 65 AMG
1352	Rover Defender X-Dynamic SE
997	MKZ Hybrid Base
752	GLA 250 Base 4MATIC
1017	Malibu LTZ
1852	i3 Base
135	840 Gran Coupe i xDrive
349	CLS-Class CLS 400 4MATIC
1835	Yukon XL AT4
1361	Rover Discovery Sport SE R-Dynamic
463	Clubman Cooper S ALL4
891	Jetta 1.4T R-Line
255	Accord Sport
420	Cayenne E-Hybrid S Platinum Edition
668	F-250 XL Crew Cab Super Duty
1005	MX-5 Miata RF Grand Touring
1376	Rover Range Rover Evoque Base
728	G 550 4x4 Squared Base
888	Integra GS-R
361	CT5-V Blackwing
384	CX-9 Carbon Edition
1199	Q3 45 S line Premium Plus
1711	V60 T6 R-Design Platinum
805	Gladiator Sport
915	LX 470 Base
744	G80 3.3T Sport
209	AMG C AMG C 63
1732	Wrangler S
1540	Silverado 1500 W/T
1206	Q5 3.0 TDI Premium Plus
1332	Rio S
7	1500 Classic SLT
1624	TT 3.2 Cabriolet quattro
1646	Taycan Base
1543	Silverado 2500 H/D
1341	Romeo Stelvio Base
1289	RSX Base
989	MKC Select
1246	Quattroporte S GranLusso
294	Beetle 2.0T Final Edition SE
1790	XC90 Hybrid T8 Inscription
481	Continental GT V8 S
335	C-Class C300 4MATIC
1355	Rover Discovery LSE
366	CTS 3.6L Luxury
1448	S60 B5 Inscription
1040	Mazda3 FWD w/Premium Package
1061	Mirai Limited
558	ES 330 Base
304	Blazer Premier
1308	Rainier CXL
760	GLE 350 GLE 350
1815	XKR Base
364	CT6-V 4.2L Blackwing Twin Turbo
646	Express 3500 LT
1642	Tahoe RST
383	CX-7 Sport
640	Explorer sport
321	Bronco Sport Base
755	GLC 300 Base
220	AMG GLC 43 AMG GLC 43
480	Continental GT V8 First Edition
1582	Sprinter 3500XD High Roof
459	Civic Si Base
263	Armada SV
958	M37 x
940	Levante S
1429	S-Class S500
1506	Sienna XLE Limited
968	M56 Base
412	Capri XR2
426	Cayenne Turbo GT
1278	RS 3 2.5T
317	Bronco Heritage Edition Advanced
1270	RDX Advance Package
1416	Rover Range Rover Velar SVAutobiography Dynamic Edition
359	CT4 Luxury
34	300C SRT8
442	Challenger SRT8 392
1661	Titan S
1822	XT6 Premium Luxury AWD
990	MKS Base
619	Excursion XLS
1231	Q8 55 Premium
1389	Rover Range Rover P530 SE
608	Escape PHEV SE
254	Accord Hybrid Touring
26	2500 Longhorn
1212	Q50 3.0T Premium
363	CT6 Luxury
552	E-PACE S
1575	Sportage S
271	Atlas 3.6L SEL Premium
1635	Tacoma TRD Sport
1629	Tacoma Base
1307	RX-8 Sport
1099	Mustang Mach-E Premium
523	DTS Base
1758	X4 M40i
1415	Rover Range Rover Velar R-Dynamic SE
298	Bentayga Onyx Edition
101	535 i
829	Grand Cherokee L Overland
1497	Seltos S
618	Excursion Limited Ultimate
1465	SL-Class SL400
991	MKS EcoBoost
1619	TSX Technology
236	ATS 2.0L Turbo
1677	Trailblazer RS
1714	Veloster Value Edition
1853	i3 Base w/Range Extender
1073	Model S P100D
1056	MazdaSpeed Miata MX-5 Base
578	Enclave Essence
1791	XC90 Hybrid T8 R-Design
1296	RX 350 F SPORT Appearance
330	C-Class C 300 4MATIC Luxury
1627	TT Roadster quattro
1201	Q4 e-tron 50 Premium Plus
944	Lucerne CXL
1527	Silverado 1500 Custom Trail Boss
87	4Runner TRD Sport
837	Grecale Modena
439	Challenger SRT Demon
860	ID.4 Pro S
85	4Runner TRD Off Road
641	Express 1500 Base
939	Levante Modena
570	Elantra GLS
484	Continental GTC V8
92	500 Sport
1845	Z4 sDrive35is
1696	Tundra Grade
1375	Rover Range Rover Autobiography
901	K5 EX
1561	Sonic LT
764	GLS 550 Base 4MATIC
856	Huracan LP580-2
1381	Rover Range Rover Evoque SE
972	M8 Base
1585	Sprinter Normal Roof
1611	TLX Tech
1818	XT4 Sport
722	Fusion Energi Titanium
1669	Touareg VR6 Lux
1456	S8 4.0T quattro
648	F-150 Heritage XL SuperCab Flareside
966	M5 Competition
1356	Rover Discovery S
1313	Ram 1500 SRT-10
1113	NX 300h Base
1209	Q5 Premium
927	Lancer Evolution IX
1579	Sprinter 2500 High Roof
1142	Outback 3.0 R VDC Limited
61	370Z NISMO Tech
1353	Rover Discovery HSE
1208	Q5 40 Premium
171	A4 2.0T Tech Premium
1551	Silverado 3500 LTZ
794	Genesis Coupe 3.8 R-Spec
1481	SQ8 4.0T Premium Plus
320	Bronco Raptor
225	AMG GLE 63 S Coupe 4MATIC
906	Kona N Base
1706	Tundra TRD Pro
782	Gallardo Base
67	430 430i
1647	Taycan Turbo
1368	Rover Range Rover 3.0L Supercharged
1137	Optima S
930	Lancer Sportback ES
89	4Runner Venture
105	550 Gran Turismo i
1457	S8 4.2 quattro
1715	Venza XLE
935	Legacy 2.5 GT spec.B
414	Carrera GT Base
1722	Vue Hybrid Base
845	Highlander Hybrid Limited
1850	i3 120Ah w/Range Extender
938	Levante GTS
388	CX-9 Touring Plus
535	Durango SRT
1038	Maybach S S 600
978	MDX 3.5L Technology Package
134	812 Superfast Base
496	Corsair Grand Touring
362	CT5-V V-Series
1272	RDX PMC Edition
471	Compass High Altitude
1653	Terrain Denali
1507	Sienna XSE 25th Anniversary
1365	Rover LR4 HSE
103	540 i
1107	NV Passenger NV3500 HD SL V8
356	CR-V Touring
1659	Tiguan 2.0T SE
154	911 GT2 RS
1303	RX 450h F Sport
1092	Mustang Bullitt
1006	MX-5 Miata Shinsen
116	718 Boxster Base
1848	allroad 2.7T
380	CX-5 Touring
1	128 i
429	Cayman GT4
1074	Model X 75D
691	F12berlinetta Base
1717	Versa 1.6 SL
1360	Rover Discovery Sport SE
1033	Maxima 3.5 SR
284	Aventador LP700-4
1521	Sierra 2500 SLE H/D Extended Cab
458	Civic Si
1186	Prelude Type SH
912	LS 500 Base
285	Aventador LP750-4 Superveloce
1132	Odyssey LX
19	200 Limited
259	Altima 2.5 SV
267	Arteon 2.0T SEL R-Line
822	GranTurismo MC
252	Accord EX-L 2.0T
1567	Sorento Plug-In Hybrid SX
1065	Model 3 Performance
1150	Outlander Sport ES
979	MDX 3.5L w/Advance & Entertainment Pkgs
54	3500 SLT
77	488 GTB Base
1050	Mazda6 Grand Touring Reserve
305	Bolt EUV Premier
1080	Model X Performance
152	911 Carrera S Cabriolet
1129	Niro Plug-In Hybrid EX Premium
1126	Niro EV EX
1546	Silverado 2500 LT H/D Extended Cab
1645	Taycan 4S
485	Continental Reserve
1554	Solstice Base
1299	RX 350 RX 350
1698	Tundra Hybrid Limited
1783	XC60 Recharge Plug-In Hybrid T8 Inscription
682	F-PACE 30t R-Sport
495	Corolla iM Base
707	Focus SEL
590	Escalade Base
1196	Prowler Base
1634	Tacoma TRD Pro
684	F-PACE S
1128	Niro Plug-In Hybrid EX
27	2500 Powerwagon
16	1500 TRX
960	M4 CS
1488	Santa Cruz 2.5T Limited
1189	Prius Two
724	Fusion Hybrid SE
1198	Q3 45 S line Premium
1670	Town & Country LX
224	AMG GLE 53 AMG GLE 53
221	AMG GLC 63 Base 4MATIC
1081	Model X Plaid
1851	i3 94 Ah
1797	XE 35t Premium
1695	Tundra 1794
1010	Macan GTS
1256	R8 5.2 V10
1589	Suburban 1500 LS
863	ILX Premium & A-SPEC Packages
1203	Q40 Base
41	325 xi
1244	Quattroporte Modena Q4
0	124 Spider Abarth
449	Charger SE
605	Escalade Sport
1823	XT6 Sport AWD
1515	Sierra 1500 SLE1 Extended Cab
1182	Pilot Black Edition
1450	S60 Recharge Plug-In Hybrid T8 Inscription
1839	Z3 3.0i Roadster
74	440 Gran Coupe 440i xDrive
601	Escalade Platinum Edition
1093	Mustang EcoBoost
786	Gallardo SE
726	Fusion Hybrid Titanium
1493	Santa Fe Sport 2.0L Turbo
1660	Tiguan 2.0T SEL Premium R-Line
874	Impala 2LZ
903	K900 5.0L
1323	Ranger Edge SuperCab
1014	Magnum R/T
201	ALPINA B7 ALPINA B7 xDrive
1114	NX 350 F SPORT Handling
348	CLS 450 Base
1549	Silverado 2500 WT
740	G70 3.3T
1528	Silverado 1500 High Country
357	CT 200h Base
343	CC Sport
1735	Wrangler Sahara Altitude
286	Aventador S Base
307	Boxster Black Edition
1509	Sierra 1500 Denali
1138	Optima SX Turbo
1034	Maxima GLE
1146	Outback Limited XT
1437	S4 4.2 quattro
1155	Palisade Limited
163	A3 2.0 TDI Premium Plus
1849	e-tron Prestige
1694	Tucson SEL
693	FF Base
1059	Mirage ES
1662	Titan SE
128	750 750i xDrive
717	Forte LX
121	718 Cayman S
1304	RX 450h F Sport Handling
1072	Model S Long Range Plus
273	Atlas Cross Sport 3.6L V6 SEL Premium
1262	RAV4 Hybrid XLE
534	Durango SLT
663	F-250 LARIAT
1079	Model X P90D
196	A8 4.0
464	Colorado LT
1336	Rogue Sport S
75	440 Gran Coupe i
1076	Model X Long Range
1592	Suburban 2500
575	Elise Base
1476	SLS AMG Base
1569	Sorento S
1379	Rover Range Rover Evoque R-Dynamic HSE
508	Corvette ZR-1
1397	Rover Range Rover Sport First Edition
661	F-250 Base
91	500 Lounge
1778	X7 M50i
1817	XLR V
177	A5 45 S line Premium Plus
498	Corsair Standard
340	C40 Recharge Pure Electric Twin Ultimate
696	FX37 Base
651	F-150 Lightning LARIAT
1062	Model 3 Base
1260	R8 Base
219	AMG GLA 45 Base 4MATIC
1676	Trailblazer LT
1838	Z Proto Spec
486	Convertible Cooper
1069	Model S 75D
982	MDX Sport Hybrid 3.0L w/Technology Package
288	Aviator Luxury
977	MC20 MC20
857	Huracan LP580-2S
1149	Outlander Sport 2.4 SE
1041	Mazda3 Grand Touring
761	GLE 450 GLE 450
354	CR-V EX-L
1529	Silverado 1500 Hybrid 1HY
1180	Phantom Drophead Coupe Drophead
904	Kicks S
638	Explorer Timberline
1496	Sebring LX
1607	TLX
686	F-TYPE
1321	Ram 3500 Quad Cab DRW
613	Evora 2+2
1100	Mustang Mach-E Select
1252	R1S Launch Edition
100	530e Base
1210	Q5 S line Premium
1346	Rover Defender
1777	X7 ALPINA XB7
859	Huracan Tecnica Coupe
1816	XLR Base
985	MDX w/Advance Package
516	Crosstrek Premium
494	Corolla S Plus
665	F-250 Limited
178	A5 Sportback S line Premium Plus
1577	Sportage X-Pro
1536	Silverado 1500 Limited Custom
588	Equinox Premier w/2LZ
1115	NX 350h Premium
1127	Niro EX
93	500X Trekking Plus
265	Arteon 2.0T SE
1351	Rover Defender X
1152	Pacifica Launch Edition
205	AMG C 43 Base
830	Grand Cherokee L Summit
880	Impreza Premium
94	500e Battery Electric
477	Continental GT GT Speed
1583	Sprinter 4500 High Roof
1254	R8 4.2 quattro Spyder
826	Grand Cherokee Altitude
1071	Model S 85D
1719	Veyron 16.4 Grand Sport
765	GR Corolla Circuit Edition
1202	Q4 e-tron Sportback Premium
509	Countryman Cooper
1418	S-10 LS
933	Leaf SL
1318	Ram 2500 SLT Mega Cab
1225	Q7 Premium Plus
1616	TLX w/A-Spec Package
90	4Runner Venture Special Edition
1780	X7 xDrive50i
1091	Mustang Boss 302
1275	RDX w/Advance Package
1070	Model S 85
455	Cherokee X
1253	R1T Launch Edition
1171	Passport Elite
1168	Passat 2.0T R-Line
1269	RDX
1286	RS 7 4.0T quattro
690	F-TYPE V6 S
866	IONIQ 5 SE
560	EV6 GT-Line
934	Leaf SV PLUS
1200	Q3 S line Premium Plus
1691	Tucson Hybrid SEL Convenience
758	GLE 350 Base
1068	Model S 70D
729	G-Class G 550
68	430 Gran Coupe i xDrive
890	Integra w/A-Spec Tech Package
212	AMG E 53 4MATIC
905	Kona EV SEL
1151	Pacifica Hybrid Touring L
709	ForTwo Pure
1728	Wraith Base
1064	Model 3 Mid Range"""
# Add your full 1859 model entries here

fuel_type_data = """2	Gasoline
1	Hybrid
0	E85 Flex Fuel
3	Diesel
4	Plug-In Hybrid
5	Hydrogen"""

ext_col_data = """28	Blue
19	Black
194	Purple
107	Gray
263	White
201	Red
226	Silver
244	Summit White
188	Platinum Quartz Metallic
108	Green
176	Orange
135	Lunar Rock
203	Red Quartz Tintcoat
13	Beige
104	Gold
126	Jet Black Mica
77	Delmonico Red Pearlcoat
38	Brown
206	Rich Garnet Metallic
240	Stellar Black Metallic
271	Yellow
74	Deep Black Pearl Effect
153	Metallic
115	Ice Silver Metallic
0	Agate Black Metallic
210	Rosso Mars Metallic
264	White Clearcoat
217	Santorini Black Metallic
61	DB Black Clearcoat
234	Snowflake White Pearl
97	Glacial White Pearl
152	Maximum Steel Metallic
29	Blue Caelum
65	Dark Matter Metallic
274	â€“
178	Oxford White
55	Cobra Beige Metallic
257	Velvet Red Pearlcoat
195	Python Green
172	Obsidian Black Metallic
14	Beluga Black
31	Blue Reflex Mica
238	Sparkling Silver
21	Black Clearcoat
237	Soul Red Crystal Metallic
34	Bright White Clearcoat
223	Shimmering Silver
155	Midnight Black Metallic
41	Cajun Red Tintcoat
54	Cirrus Silver Metallic
27	Blu
39	Burnished Bronze Metallic
110	Hellayella Clearcoat
79	Diamond Black
265	White Diamond Tri-Coat
165	Nebula Gray Pearl
37	Bronze Dune Metallic
190	Polymetal Gray Metallic
168	Nightfall Gray Metallic
93	Fuji White
222	Shadow Gray Metallic
66	Dark Moon Blue Metallic
68	Dark Sapphire
87	Firecracker Red Clearcoat
25	Black Raven
267	White Knuckle Clearcoat
232	Siren Red Tintcoat
78	Designo Magno Matte
8	Atomic Silver
18	Billet Silver Metallic Clearcoat
84	Ember Pearlcoat
139	Magnetic Black
180	Pacific Blue Metallic
156	Midnight Blue Metallic
3	Alta White
83	Eiger Grey
80	Diamond White
174	Onyx
112	Hyper Red
82	Ebony Twilight Metallic
2	Alpine White
236	Sonic Silver Metallic
62	Dark Ash Metallic
261	Volcano Grey Metallic
182	Patriot Blue Pearlcoat
175	Onyx Black
250	Titanium Silver
105	Granite Crystal Clearcoat Metallic
167	Nero Noctis
231	Silver Zynith
20	Black Cherry
116	Iconic Silver Metallic
166	Nero Daytona
71	Daytona Gray Pearl Effect
67	Dark Moss
76	Deep Crystal Blue Mica
51	China Blue
142	Magnetic Metallic
154	Midnight Black
256	Vega Blue
213	Sandstone Metallic
138	Machine Gray Metallic
40	C / C
187	Platinum Gray Metallic
228	Silver Ice Metallic
46	Carrara White Metallic
26	Black Sapphire Metallic
103	Go Mango!
266	White Frost Tri-Coat
133	Lizard Green
215	Santorin Black
252	Twilight Black
95	Gecko Pearlcoat
218	Satin Steel Metallic
117	Imperial Blue Metallic
163	Nautical Blue Pearl
242	Stone Gray Metallic
197	Quartzite Grey Metallic
145	Majestic Black Pearl
221	Shadow Black
56	Crimson Red Tintcoat
227	Silver Flare Metallic
259	Vik Black
136	Lunar Silver Metallic
148	Maroon
204	Redline Red
121	Iridescent Pearl Tricoat
36	Brilliant Silver Metallic
134	Lunar Blue Metallic
151	Matte White
247	Tan
243	Stormy Sea
253	Twilight Blue Metallic
113	Ibis White
131	Kodiak Brown Metallic
58	Crystal Black Silica
241	Sting Gray Clearcoat
192	Pristine White
183	Pearl White
6	Antimatter Blue Metallic
248	Tango Red Metallic
57	Crystal Black Pearl
207	Rift Metallic
102	Glacier White Metallic
196	Quartz White
24	Black Obsidian
35	Brilliant Black
212	Ruby Red Metallic Tinted Clearcoat
251	Tungsten Metallic
143	Magnetite Black Metallic
128	Jupiter Red
98	Glacier
216	Santorini Black
164	Navarre Blue
184	Phantom Black Pearl Effect / Black Roof
125	Isle of Man Green Metallic
235	Snowflake White Pearl Metallic
33	Brands Hatch Gray Metallic
260	Volcanic Orange
132	Liquid Platinum
12	Bayside Blue
200	Rapid Red Metallic Tinted Clearcoat
146	Majestic Plum Metallic
268	White Platinum Tri-Coat Metallic
159	Mosaic Black Metallic
50	Chalk
23	Black Noir Pearl
169	Nightfall Mica
64	Dark Gray Metallic
162	Mythos Black Metallic
32	Blueprint
198	Quicksilver Metallic
92	Frozen White
137	Lunare White Metallic
258	Verde
63	Dark Graphite Metallic
161	Mythos Black
81	Ebony Black
254	Ultra Black
120	Ingot Silver Metallic
100	Glacier Silver Metallic
109	Gun Metallic
269	Wind Chill Pearl
5	Anodized Blue Metallic
273	designo Diamond White
249	Tempest
48	Caviar
185	Phytonic Blue Metallic
158	Moonlight Cloud
72	Daytona Gray Pearl Effect w/ Black Roof
16	Bianco Isis
43	Carbonized Gray Metallic
75	Deep Blue Metallic
4	Ametrin Metallic
245	Super Black
209	Rosso Corsa
30	Blue Metallic
122	Iridium Metallic
69	Dark Slate Metallic
147	Manhattan Noir Metallic
1	Alfa White
11	Baltic Gray
170	Northsky Blue Metallic
246	Super White
42	Carbon Black Metallic
111	Horizon Blue
127	Jungle Green
49	Cayenne Red Tintcoat
186	Pink
70	Daytona Gray
96	Gentian Blue Metallic
214	Sangria Red
129	Kemora Gray Metallic
144	Magnetite Gray Metallic
173	Octane Red Pearlcoat
60	Custom Color
255	Ultra White
179	Pacific Blue
211	Ruby Flare Pearl
47	Caspian Blue
230	Silver Radiance
45	Carpathian Grey Premium Metallic
114	Ice
10	Balloon White
224	Shoreline Blue Pearl
89	Firenze Red Metallic
205	Remington Red Metallic
177	Orca Black Metallic
199	Radiant Red Metallic II
157	Mineral White
101	Glacier White
233	Snow White Pearl
123	Iridium Silver Metallic
44	Carpathian Grey
160	Mountain Air Metallic
219	Selenite Gray Metallic
106	Graphite Grey Metallic
220	Selenite Grey Metallic
130	Kinetic Blue
270	Wolf Gray
239	Star White
85	Emin White
118	Indus Silver
149	Matador Red Metallic
191	Portofino Gray
22	Black Forest Green
73	Dazzling White
90	Flame Red Clearcoat
99	Glacier Blue Metallic
7	Arctic White
208	Rosso
229	Silver Mist
15	Bianco Icarus Metallic
88	Firenze Red
124	Ironman Silver
202	Red Obsession
141	Magnetic Gray Metallic
94	Garnet Red Metallic
262	Vulcano Black Metallic
140	Magnetic Gray Clearcoat
272	Yulong
150	Matador Red Mica
189	Platinum White Pearl
119	Infrared Tintcoat
17	Billet Clearcoat Metallic
225	Silician Yellow
181	Passion Red
91	Frozen Dark Silver Metallic
59	Crystal White Pearl
193	Pure White
53	Chronos Gray Metallic
9	Aventurine Green Metallic
52	Chronos Gray
86	Eminent White Pearl
171	Obsidian"""

int_col_data = """63	Gray
12	Black
9	Beige
28	Brown
118	Silver
70	Jet Black
86	Mesa
129	White
132	â€“
103	Red
25	Blue
85	Medium Stone
7	Ash
53	Ebony
116	Shara Beige
121	Tan
124	Titan Black / Quarzit
59	Global Black
95	Orange
110	Saddle Brown
92	Nero Ade
10	Beluga
74	Light Slate
60	Gold
19	Black Onyx
93	Nougat Brown
29	Camel
67	Hotspur Hide
36	Charcoal
114	Satin Black
49	Deep Chestnut
52	Diesel Gray / Black
130	White / Brown
0	AMG Black
98	Parchment
115	Shale
31	Canberra Beige
111	Sahara Tan
55	Ebony / Pimento
106	Rhapsody Blue
81	Medium Dark Slate
107	Rioja Red
14	Black / Express Red
51	Deep Garnet
102	Portland
113	Sandstone
45	Dark Ash
50	Deep Cypress
18	Black / Stone Grey
40	Chestnut
90	Navy Pier
65	Green
58	Giallo Taurus / Nero Ade
87	Mistral Gray / Raven
48	Dark Gray
4	Amber
38	Charles Blue
66	Hotspur
82	Medium Earth Gray
35	Ceramic
71	Kyalami Orange
37	Charcoal Black
1	Adrenaline Red
127	Walnut
27	Brandy
17	Black / Saddle
94	Obsidian Black
96	Oyster W/Contrast
77	Macchiato
104	Red / Black
128	Whisper Beige
62	Graphite
131	Yellow
83	Medium Light Camel
119	Slate
91	Nero
56	Ebony Black
57	Espresso
33	Cappuccino
69	Ivory / Ebony
26	Boulder
80	Medium Ash Gray
101	Platinum
73	Light Platinum / Jet Black
11	Beluga Hide
15	Black / Graphite
32	Canberra Beige/Black
109	Rock Gray
46	Dark Auburn
3	Almond Beige
112	Sand Beige
13	Black / Brown
68	Ice
117	Silk Beige/Espresso Brown
47	Dark Galvanized
43	Cobalt Blue
61	Grace White
123	Titan Black
44	Cocoa / Dune
72	Light Gray
30	Camel Leather
75	Light Titanium
100	Pimento Red w/Ebony
120	Sport
97	Oyster/Black
23	Black/Red
54	Ebony / Ebony Accents
42	Cloud
64	Graystone
8	BLACK
89	Mountain Brown
76	Linen
34	Caramel
16	Black / Gray
24	Blk
20	Black w/Red Stitching
88	Mocha
125	Tupelo
122	Tan/Ebony
99	Parchment.
22	Black/Gun Metal
126	Very Light Cashmere
84	Medium Pewter
105	Red/Black
6	Aragon Brown
21	Black/Graphite
5	Anthracite
41	Classic Red
79	Magma Red
108	Roast
78	Macchiato/Magmagrey
2	Agave Green
39	Chateau"""

accident_data = """1	None reported
0	At least 1 accident or damage reported"""

automatic_data = """0	Manual
1	Automatic
-1	Unknown
2	Transmission w/Dual Shift Mode
3	CVT Transmission
4	Transmission Overdrive Switch
5	CVT-F
6	Variable
7	F"""

cylinder_info_data = """0	Unknown
1	Unknown with Rotary engine
2	10 Cylinders
3	12 Cylinders
4	5 Cylinders
5	8 Cylinders
6	EV vehicle
7	H4
8	H6
9	I3
10	I4
11	I6
12	V10
13	V12
14	V6
15	V8
16	W12
17	W16"""

engine_type_data = """1	Cylinder Engine
5	Electric Motor
10	GDI DOHC Turbo
43	PDI DOHC Twin Turbo
0	-1
30	MPFI DOHC
40	PDI DOHC
36	MPFI OHV
6	GDI DOHC
9	GDI DOHC Supercharged
13	GDI DOHC Twin Turbo
14	GDI OHV
37	MPFI OHV Flexible Fuel
35	MPFI DOHC Twin Turbo
42	PDI DOHC Turbo
16	GDI SOHC
45	Rotary engine
31	MPFI DOHC Flexible Fuel
12	GDI DOHC Turbo Hybrid
28	Liter Turbo
22	Liter GTDI
2	DDI DOHC Turbo Diesel
17	GDI SOHC Turbo
41	PDI DOHC Hybrid
20	Liter DOHC Twin Turbo
34	MPFI DOHC Turbo
4	DDI OHV Twin Turbo Diesel
24	Liter SC ULEV
15	GDI OHV Supercharged
3	DDI OHV Turbo Diesel
39	MPFI SOHC Flexible Fuel
46	SOHC I-VTEC V6
29	Liter Twin Turbo
26	Liter TFSI
44	PDI DOHC Twin Turbo Hybrid
11	GDI DOHC Turbo Flexible Fuel
8	GDI DOHC Hybrid
38	MPFI SOHC
18	Intercooled Turbo Diesel
23	Liter SC DOHC
7	GDI DOHC Flexible Fuel
21	Liter GDI DOHC Twin Turbo
32	MPFI DOHC Hybrid
47	Standard Range Battery
19	Liter DOHC
33	MPFI DOHC Supercharged
27	Liter TSI
25	Liter Supercharged"""

# Parse all mappings into dictionaries
mappings = {
    'brand': parse_mapping(brand_data),
    'model': parse_mapping(model_data),
    'fuel_type': parse_mapping(fuel_type_data),
    'ext_col': parse_mapping(ext_col_data),
    'int_col': parse_mapping(int_col_data),
    'accident': parse_mapping(accident_data),
    'automatic': parse_mapping(automatic_data),
    'Cylinder Info': parse_mapping(cylinder_info_data),
    'engine type': parse_mapping(engine_type_data)
}

# Load your trained model (update the path to your actual model file)
model = joblib.load('car_price_model_new.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect and encode categorical features
            brand = mappings['brand'][request.form['brand']]
            model_val = mappings['model'][request.form['model']]
            fuel_type = mappings['fuel_type'][request.form['fuel_type']]
            ext_col = mappings['ext_col'][request.form['ext_col']]
            int_col = mappings['int_col'][request.form['int_col']]
            accident = mappings['accident'][request.form['accident']]
            automatic = mappings['automatic'][request.form['automatic']]
            cylinder_info = mappings['Cylinder Info'][request.form['cylinder_info']]
            engine_type = mappings['engine type'][request.form['engine_type']]

            # Collect numerical features and convert to appropriate types
            model_year = int(request.form['model_year'])
            milage = float(request.form['milage'])
            litre = float(request.form['litre'])
            hp = float(request.form['hp'])
            speed = float(request.form['speed'])
            voltage = float(request.form['voltage'])

            # Arrange inputs in the exact order your model was trained on
            input_data = np.array([[brand, model_val, model_year, milage, fuel_type, ext_col, int_col,
                                    accident, litre, automatic, hp, speed, cylinder_info, engine_type, voltage]])

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Render the form again with the prediction
            return render_template('index.html',
                                   prediction=prediction,
                                   brands=sorted(mappings['brand'].keys()),
                                   models=sorted(mappings['model'].keys()),
                                   fuel_types=sorted(mappings['fuel_type'].keys()),
                                   ext_cols=sorted(mappings['ext_col'].keys()),
                                   int_cols=sorted(mappings['int_col'].keys()),
                                   accidents=sorted(mappings['accident'].keys()),
                                   automatics=sorted(mappings['automatic'].keys()),
                                   cylinder_infos=sorted(mappings['Cylinder Info'].keys()),
                                   engine_types=sorted(mappings['engine type'].keys()))

        except KeyError as e:
            return f"Error: Invalid selection for {e}. Please choose a valid option."
        except ValueError as e:
            return f"Error: Invalid input value. {e}"

    # GET request: render the empty form
    return render_template('index.html',
                           brands=sorted(mappings['brand'].keys()),
                           models=sorted(mappings['model'].keys()),
                           fuel_types=sorted(mappings['fuel_type'].keys()),
                           ext_cols=sorted(mappings['ext_col'].keys()),
                           int_cols=sorted(mappings['int_col'].keys()),
                           accidents=sorted(mappings['accident'].keys()),
                           automatics=sorted(mappings['automatic'].keys()),
                           cylinder_infos=sorted(mappings['Cylinder Info'].keys()),
                           engine_types=sorted(mappings['engine type'].keys()))

if __name__ == '__main__':
    app.run(debug=True)