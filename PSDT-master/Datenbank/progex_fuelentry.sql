-- MySQL dump 10.13  Distrib 5.7.17, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: progex
-- ------------------------------------------------------
-- Server version	5.7.18-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `fuelentry`
--

DROP TABLE IF EXISTS `fuelentry`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `fuelentry` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `carID` int(11) NOT NULL,
  `daytime` varchar(20) NOT NULL,
  `ppLiter` double NOT NULL,
  `endcost` double NOT NULL,
  `userid` int(11) NOT NULL,
  `fuelID` int(11) NOT NULL,
  `photo` mediumblob,
  `kmstatus` int(11) NOT NULL,
  `literamount` double NOT NULL,
  `stationname` varchar(20) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fuelID` (`fuelID`),
  KEY `userid` (`userid`),
  KEY `carID` (`carID`),
  CONSTRAINT `fuelentry_ibfk_1` FOREIGN KEY (`fuelID`) REFERENCES `fuel` (`id`),
  CONSTRAINT `fuelentry_ibfk_2` FOREIGN KEY (`userid`) REFERENCES `userdata` (`id`),
  CONSTRAINT `fuelentry_ibfk_3` FOREIGN KEY (`carID`) REFERENCES `car` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `fuelentry`
--

LOCK TABLES `fuelentry` WRITE;
/*!40000 ALTER TABLE `fuelentry` DISABLE KEYS */;
INSERT INTO `fuelentry` VALUES (1,3,'2017-05-01 00:00:00',1.31,75.23,1,1,NULL,100100,57.43,'Shell'),(2,3,'2017-05-02 00:00:00',1.29,51.6,1,1,NULL,100200,40,'Shell'),(3,4,'2017-05-05 00:00:00',1.36,81.6,1,1,NULL,120100,60,'Shell'),(4,4,'2017-05-03 00:00:00',1.24,24.8,1,1,NULL,120150,20,'Shell'),(5,3,'2017-05-03 00:00:00',1.2,54,1,1,NULL,100250,45,'Shell'),(6,3,'2017-05-10 00:00:00',1.33,39.9,1,1,NULL,100300,30,'Shell'),(7,3,'2017-05-13 00:00:00',1.15,92,1,1,NULL,100450,80,'Shell'),(8,3,'2017-06-01 00:00:00',1.13,73.45,1,1,NULL,100500,65,'Shell'),(10,3,'2017-06-45',1.14,74.47,1,1,NULL,90200,65.33,'Shell'),(11,9,'2017-06-45',1.14,74.47,1,1,NULL,90200,65.33,'Shell'),(12,9,'2017-06-22',2.5,50,1,1,NULL,90250,20,'Shell');
/*!40000 ALTER TABLE `fuelentry` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2017-06-23 17:18:25
