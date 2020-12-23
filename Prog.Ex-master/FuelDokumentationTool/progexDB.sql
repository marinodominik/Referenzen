-- MySQL dump 10.13  Distrib 5.7.12, for osx10.9 (x86_64)
--
-- Host: 127.0.0.1    Database: ProgEx
-- ------------------------------------------------------
-- Server version	5.7.16

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
-- Table structure for table `CAR`
--

DROP TABLE IF EXISTS `CAR`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `CAR` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `CarName` varchar(30) NOT NULL,
  `kmstatus` varchar(20) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `CAR`
--

LOCK TABLES `CAR` WRITE;
/*!40000 ALTER TABLE `CAR` DISABLE KEYS */;
INSERT INTO `CAR` VALUES (1,'Mercedes CL63 AMG','25000');
/*!40000 ALTER TABLE `CAR` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `FUEL`
--

DROP TABLE IF EXISTS `FUEL`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `FUEL` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `fuelname` varchar(60) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `FUEL`
--

LOCK TABLES `FUEL` WRITE;
/*!40000 ALTER TABLE `FUEL` DISABLE KEYS */;
INSERT INTO `FUEL` VALUES (1,'Super 95'),(2,'Diesel'),(3,'LPG'),(4,'Super E10'),(5,'SuperPlus'),(6,'DieselPlus');
/*!40000 ALTER TABLE `FUEL` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `FuelEntry`
--

DROP TABLE IF EXISTS `FuelEntry`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `FuelEntry` (
  `id` Integer(11) NOT NULL AUTO_INCREMENT,
  `carID` int(11) NOT NULL,
  `stationname` varchar(60) DEFAULT NULL,
  `daytime` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `ppLiter` double NOT NULL,
  `literamount` double NOT NULL,
  `endcost` double NOT NULL,
  `userid` int(11) NOT NULL,
  `fuelID` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `car2car` (`carID`),
  KEY `user2user` (`userid`),
  KEY `fuel2fuel` (`fuelID`),
  CONSTRAINT `car2car` FOREIGN KEY (`carID`) REFERENCES `CAR` (`id`),
  CONSTRAINT `fuel2fuel` FOREIGN KEY (`fuelID`) REFERENCES `FUEL` (`id`),
  CONSTRAINT `user2user` FOREIGN KEY (`userid`) REFERENCES `UserData` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `FuelEntry`
--

LOCK TABLES `FuelEntry` WRITE;
/*!40000 ALTER TABLE `FuelEntry` DISABLE KEYS */;
INSERT INTO `FuelEntry` VALUES (1,1,'SHELL','2023-05-20 15:00:00',1.54,20,30.8,1,5);
/*!40000 ALTER TABLE `FuelEntry` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `UserData`
--

DROP TABLE IF EXISTS `UserData`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `UserData` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `firstname` varchar(60) NOT NULL,
  `lastname` varchar(60) NOT NULL,
  `Gender` varchar(10) NOT NULL,
  `birth` varchar(10) NOT NULL,
  `username` varchar(30) NOT NULL,
  `userpassword` varchar(60) NOT NULL,
  `mobilnr` varchar(20) NOT NULL,
  `adress` varchar(60) NOT NULL,
  `city` varchar(60) NOT NULL,
  `zip` varchar(10) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `UserData`
--

LOCK TABLES `UserData` WRITE;
/*!40000 ALTER TABLE `UserData` DISABLE KEYS */;
INSERT INTO `UserData` VALUES (1,'Emre','Dogan','Male','17.09.1995','emredogan','emredogan','0123456789','Teststraße','Testhausen','65795'),(2,'firstname','lastname','gender','birth','username','userpassword','mobilrn','adress','city','zip'),(4,'Test','test','test','test','test','test','test','test','test','test'),(5,'firstname','lastname','gender','birth','username','userpassword','mobilrn','adress','city','zip'),(6,'h','h','h','h','h','h','h','h','h','h'),(7,'','','','','','','','','',''),(8,'dominik1','marino','male','01.01.2000','dominikmarino','dominik','012345','teststraße','testhausen','12345');
/*!40000 ALTER TABLE `UserData` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping events for database 'ProgEx'
--

--
-- Dumping routines for database 'ProgEx'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2017-05-30 17:39:20
